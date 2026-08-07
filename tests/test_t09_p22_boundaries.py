"""Exercise P22 trust boundaries with production T08 and T09 APIs.

These focused tests model the current multi-document Ingest producer shape,
canonical-byte ownership, resolver ownership, gold-closure reachability, and
direct public-record construction. They use configured local Storage and the
pinned public Ingest package, never a sibling checkout, parser, source reparse,
model transport, or modified frozen fixture.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
from cognityx_ingest import (
    CanonicalContentArtifact,
    EvidenceSetAddress,
    NodeSpan,
    ProvenanceAddressResolver,
    ProvenanceTarget,
    SegmentationViewService,
    SourceGraphBuilder,
    StrongProvenanceAddress,
    build_strong_address_catalog,
)
from cognityx_storage import (
    LocalStorageBackend,
    StorageClient,
    StorageConfig,
    StorageRuntime,
)

from cognityx_dataforge.handoff import (
    CompositeKnowledgeUnitHandoff,
    DataForgeHandoffService,
    GeneratedQuestionAnswer,
    GoldRelationClosure,
    HandoffArtifactStore,
    HandoffPersistenceError,
    HandoffSupportError,
    HandoffValidationError,
    ParagraphHandoffInput,
    ParagraphQAHandoff,
    SupportedClaim,
    ValidatedEvidenceBundle,
)
from cognityx_dataforge.source import (
    ResolvedV32Document,
    ResolvedV32SourceBundle,
    V32HandoffUnavailableError,
    V32SourceConflictError,
    resolve_source,
)
from t09_support import (
    FrozenClaimGenerator,
    FrozenQuestionAnswerGenerator,
    foreign_canonical_with_same_node_ids,
    frozen_evidence_bundle,
    frozen_paragraph_view,
    split_frozen_canonical_artifacts,
)


class _RecordingClaimGenerator:
    """Record calls so reachability failures can prove generation never starts.

    Adversarial composite tests construct this sequential fake. It returns a
    deterministic generated string if called, performs no I/O, and intentionally
    owns no support selection or concurrency guarantees beyond one test thread.
    """

    def __init__(self) -> None:
        """Start one isolated test fake with an empty mutable call ledger."""
        self.calls: list[tuple[str, str]] = []

    def generate_claim(self, claim_id: str, content: str) -> str:
        """Record transient inputs and return bounded generated claim text."""
        self.calls.append((claim_id, content))
        return f"Generated claim for {claim_id}"


class _StringLike:
    """Expose ``strip`` without being a string for trust-boundary testing."""

    def strip(self) -> str:
        """Return text only to prove duck typing is intentionally insufficient."""
        return "not-a-real-string"


def _put_ingest_shaped_run(
    tmp_path: Path,
) -> tuple[
    StorageRuntime,
    str,
    tuple[CanonicalContentArtifact, CanonicalContentArtifact],
    tuple[dict[str, str], dict[str, str]],
]:
    """Persist a two-document producer-compatible run with independent T08 facts.

    The integration test calls this once. For each frozen resource the helper
    builds an independent graph/catalog through public T08 builders, writes exact
    canonical/graph/catalog/provenance bytes through configured Storage, and
    emits direct references in policy-then-authority producer order. Advertised
    parser and source objects remain absent, so successful consumption proves
    they were not read. The helper returns runtime, manifest URI, canonical
    artifacts, and URI records; temporary Storage owns all lifecycle and locking.
    """
    runtime = StorageRuntime.from_config(
        StorageConfig.built_in(root=tmp_path / "storage")
    )
    artifact_store = runtime.for_role("artifact")
    canonicals = split_frozen_canonical_artifacts()
    references: list[dict[str, str]] = []
    provenance_uris: list[str] = []
    source_assets: list[dict[str, str]] = []
    for canonical in canonicals:
        graph = SourceGraphBuilder().build((canonical,))
        catalog = build_strong_address_catalog(graph, (canonical,))
        prefix = f"ingest/{canonical.document_id}"
        canonical_uri = str(
            artifact_store.put_bytes(
                f"{prefix}/canonical-content.json",
                canonical.to_json_bytes(),
                media_type="application/json",
            ).uri
        )
        graph_uri = str(
            artifact_store.put_bytes(
                f"{prefix}/source-graph.json",
                graph.to_json_bytes(),
                media_type="application/json",
            ).uri
        )
        address_uri = str(
            artifact_store.put_bytes(
                f"{prefix}/provenance-addresses.json",
                catalog.to_json_bytes(),
                media_type="application/json",
            ).uri
        )
        parser_uri = f"storage://shared/{prefix}/parser/native.json"
        source_uri = f"storage://shared/{prefix}/source/original.bin"
        artifact_uris: dict[str, object] = {
            "canonical_content": canonical_uri,
            "source_graph": graph_uri,
            "provenance_addresses": address_uri,
            "parser": {"never-open": parser_uri},
            "source": source_uri,
        }
        provenance_uri = str(
            artifact_store.put_json(
                f"{prefix}/provenance.json",
                {
                    "schema": "cognityx.ingest.provenance",
                    "schema_version": "cognityx.ingest.provenance/v2",
                    "document_id": canonical.document_id,
                    "source_asset": {
                        "asset_id": canonical.document_id,
                        "blob_sha256": canonical.resources[0].source_sha256,
                    },
                    "artifact_uris": artifact_uris,
                    "evidence": [],
                },
            ).uri
        )
        reference = {
            "document_id": canonical.document_id,
            "provenance_uri": provenance_uri,
            "canonical_content_uri": canonical_uri,
            "source_graph_uri": graph_uri,
            "provenance_addresses_uri": address_uri,
        }
        references.append(reference)
        provenance_uris.append(provenance_uri)
        source_assets.append(
            {
                "asset_id": canonical.document_id,
                "sha256": canonical.resources[0].source_sha256,
            }
        )
    manifest_uri = str(
        artifact_store.put_json(
            "ingest/run-p22/manifest.json",
            {
                "schema": "cognityx.ingest.run",
                "run_id": "run-p22",
                "context_id": "context-p22",
                "source_assets": source_assets,
                "document_ids": [item.document_id for item in canonicals],
                "evidence_refs": [],
                "provenance_refs": provenance_uris,
                "dataforge_source_refs": references,
            },
        ).uri
    )
    return runtime, manifest_uri, canonicals, (references[0], references[1])


def test_normal_multi_document_run_supports_per_document_handoff(
    tmp_path: Path,
) -> None:
    """Resolve two independent producer documents and compose each paragraph.

    This is the production-shape integration proof for P22. It preserves source
    order, loads each document's own three T08 artifacts, composes a paragraph
    through the public handoff service, leaves sibling/parser/source objects
    unopened, and rejects only the unrelated aggregate composite operation.
    """
    runtime, manifest_uri, canonicals, references = _put_ingest_shaped_run(tmp_path)
    source = resolve_source(runtime, manifest_uri).require_v3_2()
    assert tuple(item.document_id for item in source.documents) == (
        "document-policy",
        "document-authority",
    )

    for document, canonical, reference in zip(
        source.documents, canonicals, references, strict=True
    ):
        loaded = ValidatedEvidenceBundle.load_document(
            runtime, source, document.document_id
        )
        assert loaded.canonical_contents == (canonical,)
        assert loaded.canonical_content_sha256s == (
            hashlib.sha256(canonical.to_json_bytes()).hexdigest(),
        )
        segmentation = SegmentationViewService.from_canonical(canonical)
        view = segmentation.build_paragraph(view_id=f"view-{document.document_id}")
        view_set = segmentation.build_view_set((view,))
        result = DataForgeHandoffService().build_paragraph_qa(
            evidence=loaded,
            segmentation_service=segmentation,
            view_set=view_set,
            view_id=view.view_id,
            segment_id=view.segments[0].segment_id,
            generator=FrozenQuestionAnswerGenerator(),
        )
        assert result.input.node_spans == view.segments[0].node_spans
        artifact_store = runtime.for_role("artifact")
        prefix = f"ingest/{document.document_id}"
        assert not artifact_store.exists(f"{prefix}/parser/native.json")
        assert not artifact_store.exists(f"{prefix}/source/original.bin")
        assert document.to_dict() == reference

    with pytest.raises(HandoffValidationError, match="load_document.*connected"):
        ValidatedEvidenceBundle.load(runtime, source)


def test_shared_cross_resource_graph_supports_aggregate_loading(tmp_path: Path) -> None:
    """Load multiple canonical documents against one genuine connected graph.

    Aggregate composite callers use this supported path. The test builds one
    graph/catalog over both canonical documents, stores one shared byte pair,
    preserves source order, and proves the existing aggregate loader remains
    functional without merging independent facts or opening provenance payloads.
    """
    runtime = StorageRuntime.from_config(
        StorageConfig.built_in(root=tmp_path / "shared-storage")
    )
    artifact_store = runtime.for_role("artifact")
    canonicals = split_frozen_canonical_artifacts()
    base_graph = SourceGraphBuilder().build(canonicals)
    graph = replace(
        base_graph,
        graph_revision="sg-p22-shared-connected",
        relations=(
            next(
                item
                for item in frozen_evidence_bundle().source_graph.relations
                if item.relation_id == "rel-policy-to-authority"
            ),
        ),
    )
    graph.validate()
    catalog = build_strong_address_catalog(graph, canonicals)
    graph_uri = str(
        artifact_store.put_bytes(
            "shared/source-graph.json",
            graph.to_json_bytes(),
            media_type="application/json",
        ).uri
    )
    catalog_uri = str(
        artifact_store.put_bytes(
            "shared/provenance-addresses.json",
            catalog.to_json_bytes(),
            media_type="application/json",
        ).uri
    )
    documents = []
    for canonical in canonicals:
        canonical_uri = str(
            artifact_store.put_bytes(
                f"shared/{canonical.document_id}/canonical-content.json",
                canonical.to_json_bytes(),
                media_type="application/json",
            ).uri
        )
        documents.append(
            ResolvedV32Document(
                document_id=canonical.document_id,
                provenance_uri=f"storage://shared/{canonical.document_id}/provenance.json",
                canonical_content_uri=canonical_uri,
                source_graph_uri=graph_uri,
                provenance_addresses_uri=catalog_uri,
            )
        )
    loaded = ValidatedEvidenceBundle.load(
        runtime, ResolvedV32SourceBundle(tuple(documents))
    )
    assert loaded.canonical_contents == canonicals
    assert loaded.source_graph.to_json_bytes() == graph.to_json_bytes()
    assert "rel-policy-to-authority" in {
        item.relation_id for item in loaded.source_graph.relations
    }


def test_document_selection_duplicate_and_unknown_ids_fail_typed() -> None:
    """Harden direct source aggregates without sorting or fuzzy identity lookup."""
    first = ResolvedV32Document(
        document_id="document-a",
        provenance_uri="storage://shared/a/provenance.json",
        canonical_content_uri="storage://shared/a/canonical.json",
        source_graph_uri="storage://shared/a/graph.json",
        provenance_addresses_uri="storage://shared/a/addresses.json",
    )
    with pytest.raises(V32SourceConflictError, match="repeats"):
        ResolvedV32SourceBundle((first, first))
    source = ResolvedV32SourceBundle((first,))
    with pytest.raises(V32HandoffUnavailableError, match="unknown"):
        source.document("unknown")
    with pytest.raises(V32SourceConflictError, match="logical Storage URI"):
        replace(first, canonical_content_uri="file:///tmp/canonical.json")
    with pytest.raises(V32SourceConflictError, match="document_id"):
        replace(first, document_id="")


def test_foreign_same_id_canonical_view_fails_before_generation() -> None:
    """Reject foreign canonical bytes even when every segment node ID matches."""
    evidence = frozen_evidence_bundle()
    foreign = foreign_canonical_with_same_node_ids()
    segmentation, view_set = frozen_paragraph_view(foreign)
    generator = FrozenQuestionAnswerGenerator()
    assert any(item.node_id == "pol-p2" for item in foreign.content_nodes)
    with pytest.raises(HandoffValidationError, match="canonical digest"):
        DataForgeHandoffService().build_paragraph_qa(
            evidence=evidence,
            segmentation_service=segmentation,
            view_set=view_set,
            view_id="view-paragraph-v1",
            segment_id="para-2",
            generator=generator,
        )
    assert generator.calls == []


def test_exact_resolver_binding_passes_and_foreign_graph_fails_first() -> None:
    """Accept byte-equivalent resolver facts and reject a same-ID foreign graph."""
    evidence = frozen_evidence_bundle()
    segmentation, view_set = frozen_paragraph_view()
    exact_generator = FrozenQuestionAnswerGenerator()
    DataForgeHandoffService().build_paragraph_qa(
        evidence=evidence,
        segmentation_service=segmentation,
        view_set=view_set,
        view_id="view-paragraph-v1",
        segment_id="para-2",
        generator=exact_generator,
        resolver=ProvenanceAddressResolver(
            evidence.source_graph, evidence.provenance_addresses
        ),
    )
    assert len(exact_generator.calls) == 1

    foreign_revision = "sg-foreign-same-address-ids"
    foreign_graph = replace(
        evidence.source_graph, graph_revision=foreign_revision
    )
    foreign_catalog = replace(
        evidence.provenance_addresses,
        strong_addresses=tuple(
            replace(item, graph_revision=foreign_revision)
            for item in evidence.provenance_addresses.strong_addresses
        ),
    )
    generator = FrozenQuestionAnswerGenerator()
    with pytest.raises(HandoffValidationError, match="Source Graph"):
        DataForgeHandoffService().build_paragraph_qa(
            evidence=evidence,
            segmentation_service=segmentation,
            view_set=view_set,
            view_id="view-paragraph-v1",
            segment_id="para-2",
            generator=generator,
            resolver=ProvenanceAddressResolver(foreign_graph, foreign_catalog),
        )
    assert generator.calls == []


def test_foreign_catalog_with_same_strong_id_fails_before_generation() -> None:
    """Reject a byte-different catalog even when its support ID resolves exact."""
    evidence = frozen_evidence_bundle()
    segmentation, view_set = frozen_paragraph_view()
    first = evidence.provenance_addresses.strong_addresses[0]
    foreign_catalog = replace(
        evidence.provenance_addresses,
        strong_addresses=(
            replace(first, selectors=()),
            *evidence.provenance_addresses.strong_addresses[1:],
        ),
    )
    generator = FrozenQuestionAnswerGenerator()
    with pytest.raises(HandoffValidationError, match="address catalog"):
        DataForgeHandoffService().build_paragraph_qa(
            evidence=evidence,
            segmentation_service=segmentation,
            view_set=view_set,
            view_id="view-paragraph-v1",
            segment_id="para-2",
            generator=generator,
            resolver=ProvenanceAddressResolver(
                evidence.source_graph, foreign_catalog
            ),
        )
    assert generator.calls == []


def test_exact_but_unreachable_support_cannot_produce_composite() -> None:
    """Reject exact support outside the seed's explicit gold relation closure."""
    evidence = frozen_evidence_bundle()
    resource = next(
        item
        for item in evidence.source_graph.resources
        if item.resource_id == "res-authority-v2"
    )
    unrelated = StrongProvenanceAddress(
        address_id="addr-strong-unreachable-authority-root",
        source_sha256=resource.source_sha256,
        graph_revision=evidence.source_graph.graph_revision,
        resource_id=resource.resource_id,
        canonical_target=ProvenanceTarget(node_id="auth-heading-root"),
        selectors=(),
    )
    catalog = replace(
        evidence.provenance_addresses,
        strong_addresses=(
            *evidence.provenance_addresses.strong_addresses,
            unrelated,
        ),
    )
    with_unrelated = ValidatedEvidenceBundle.create(
        evidence.canonical_contents, evidence.source_graph, catalog
    )
    intent = EvidenceSetAddress(
        address_id="derived-unreachable-set",
        claim_ids=("claim-unreachable",),
        member_address_ids=(unrelated.address_id,),
    )
    generator = _RecordingClaimGenerator()
    with pytest.raises(HandoffSupportError, match="outside.*gold closure"):
        DataForgeHandoffService().build_composite_ku(
            evidence=with_unrelated,
            seed_division_id="div-policy-4.2",
            evidence_set_address_id=intent.address_id,
            evidence_set_intent=intent,
            knowledge_unit_id="ku-unreachable",
            task_schema="travel-expense-approval",
            generator=generator,
        )
    assert generator.calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gold_support_contains_only_validated_relations", False),
        ("source_reparse_allowed", True),
        ("must_not_store_independent_source_copy", False),
    ],
)
def test_composite_safety_booleans_reject_contradictory_construction(
    field: str, value: bool
) -> None:
    """Require exact immutable composite safety booleans at construction time."""
    claim = SupportedClaim(
        claim_id="claim-1",
        text="Generated claim",
        support_address_ids=("address-1",),
    )
    values: dict[str, object] = {
        "knowledge_unit_id": "ku-1",
        "task_schema": "task-1",
        "claims": (claim,),
        "evidence_set_address_id": "set-1",
        "allowed_relation_ids": ("relation-1",),
        "excluded_relation_ids": ("relation-2",),
        field: value,
    }
    with pytest.raises(HandoffValidationError):
        CompositeKnowledgeUnitHandoff(**values)


def test_public_handoff_records_reject_contradictory_direct_values() -> None:
    """Make future deserialization fail closed without relying on the service."""
    paragraph_input = ParagraphHandoffInput(
        source_graph_revision="sg-1",
        segmentation_view_id="view-1",
        segment_id="segment-1",
        node_spans=(NodeSpan(node_id="node-1"),),
    )
    with pytest.raises(HandoffValidationError, match="no-copy"):
        ParagraphQAHandoff(
            input=paragraph_input,
            question="Generated question?",
            answer="Generated answer.",
            support_address_ids=("address-1",),
            must_not_store_independent_source_copy=False,
        )
    with pytest.raises(HandoffValidationError, match="question"):
        GeneratedQuestionAnswer(question=_StringLike(), answer="answer")  # type: ignore[arg-type]
    with pytest.raises(HandoffValidationError, match="duplicate"):
        SupportedClaim(
            claim_id="claim-1",
            text="Generated claim",
            support_address_ids=("address-1", "address-1"),
        )
    with pytest.raises(HandoffValidationError, match="both"):
        GoldRelationClosure(
            seed_division_id="division-1",
            allowed_relation_ids=("relation-1",),
            excluded_relation_ids=("relation-1",),
        )


def test_handoff_store_rejects_unsupported_objects_cleanly(tmp_path: Path) -> None:
    """Return the public persistence error instead of implicit KU classification."""
    storage = StorageClient(LocalStorageBackend(tmp_path / "store")).for_shared_data()
    with pytest.raises(HandoffPersistenceError, match="unsupported"):
        HandoffArtifactStore(storage).put("unsupported-object", object())  # type: ignore[arg-type]
