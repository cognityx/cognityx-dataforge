"""Production-facing paragraph, composite, validation, and persistence tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
from cognityx_ingest import (
    EvidenceSetAddress,
    ProvenanceAddressCatalog,
    ProvenanceAddressResolver,
    SegmentationViewSet,
    SourceGraphBuilder,
    build_strong_address_catalog,
)
from cognityx_storage import (
    LocalStorageBackend,
    StorageClient,
    StorageConfig,
    StorageRuntime,
)

from cognityx_dataforge.handoff import (
    COMPOSITE_KU_HANDOFF_SCHEMA,
    PARAGRAPH_QA_HANDOFF_SCHEMA,
    DataForgeHandoffService,
    HandoffArtifactStore,
    HandoffPersistenceError,
    HandoffSupportError,
    HandoffValidationError,
    ValidatedEvidenceBundle,
    build_gold_relation_closure,
)
from cognityx_dataforge.recipes import SUPPORTED_RECIPES
from cognityx_dataforge.source import ResolvedV32Document, ResolvedV32SourceBundle
from t09_support import (
    FrozenClaimGenerator,
    FrozenQuestionAnswerGenerator,
    frozen_canonical_artifact,
    frozen_evidence_bundle,
    frozen_paragraph_view,
    load_fixture,
)


class _DenyAll:
    """Deny known strong addresses so handoff tests exercise forbidden support."""

    def allows(self, address_id: str, resource_id: str) -> bool:
        """Return false deterministically without inspecting target content."""
        return False


def _paragraph_result():
    """Build the frozen paragraph proof through production handoff composition.

    Several focused tests share this deterministic helper. It creates independent
    immutable evidence/view objects and a fake generator, performs no model,
    parser, source, or Storage call, and returns the production output plus the
    generator's transient call record.
    """
    evidence = frozen_evidence_bundle()
    segmentation, view_set = frozen_paragraph_view()
    generator = FrozenQuestionAnswerGenerator()
    result = DataForgeHandoffService().build_paragraph_qa(
        evidence=evidence,
        segmentation_service=segmentation,
        view_set=view_set,
        view_id="view-paragraph-v1",
        segment_id="para-2",
        generator=generator,
    )
    return result, generator


def test_strict_public_t08_artifact_loading_from_storage(tmp_path: Path) -> None:
    """Load production canonical/graph/address schemas with no parser artifact."""
    runtime = StorageRuntime.from_config(
        StorageConfig.built_in(root=tmp_path / "storage")
    )
    artifact_store = runtime.for_role("artifact")
    canonical = frozen_canonical_artifact()
    graph = SourceGraphBuilder().build((canonical,))
    catalog = build_strong_address_catalog(graph, (canonical,))
    canonical_uri = artifact_store.put_bytes(
        "t09/canonical.json", canonical.to_json_bytes(), media_type="application/json"
    ).uri
    graph_uri = artifact_store.put_bytes(
        "t09/graph.json", graph.to_json_bytes(), media_type="application/json"
    ).uri
    addresses_uri = artifact_store.put_bytes(
        "t09/addresses.json", catalog.to_json_bytes(), media_type="application/json"
    ).uri
    source = ResolvedV32SourceBundle(
        (
            ResolvedV32Document(
                document_id="fixture-v3-2",
                provenance_uri="storage://shared/t09/provenance.json",
                canonical_content_uri=canonical_uri,
                source_graph_uri=graph_uri,
                provenance_addresses_uri=addresses_uri,
            ),
        )
    )

    loaded = ValidatedEvidenceBundle.load(runtime, source)
    assert loaded.source_graph.graph_revision == graph.graph_revision
    assert loaded.canonical_node_text("pol-p2").startswith("The ordinary approval")
    assert not artifact_store.exists("t09/parser/raw.json")


def test_bundle_rejects_resource_hash_and_address_revision_mismatches() -> None:
    """Fail closed instead of combining or repairing incompatible T08 artifacts."""
    evidence = frozen_evidence_bundle()
    bad_resource = replace(
        evidence.source_graph.resources[0], source_sha256="0" * 64
    )
    bad_graph = replace(
        evidence.source_graph,
        resources=(bad_resource, *evidence.source_graph.resources[1:]),
    )
    with pytest.raises(HandoffValidationError, match="SHA-256"):
        ValidatedEvidenceBundle.create(
            evidence.canonical_contents, bad_graph, evidence.provenance_addresses
        )

    bad_address = replace(
        evidence.provenance_addresses.strong_addresses[0],
        graph_revision="different-revision",
    )
    bad_catalog = replace(
        evidence.provenance_addresses,
        strong_addresses=(
            bad_address,
            *evidence.provenance_addresses.strong_addresses[1:],
        ),
    )
    with pytest.raises(HandoffValidationError, match="revision"):
        ValidatedEvidenceBundle.create(
            evidence.canonical_contents, evidence.source_graph, bad_catalog
        )


def test_paragraph_handoff_matches_frozen_qa_and_exact_support() -> None:
    """Map para-2 to pol-p2 and its sole exact strong address."""
    result, generator = _paragraph_result()
    expected = load_fixture("dataforge/paragraph_qa_contract.json")

    assert result.schema == PARAGRAPH_QA_HANDOFF_SCHEMA == expected["schema"]
    assert result.input.to_dict() == expected["input"]
    assert {
        "question": result.question,
        "answer": result.answer,
        "support_address_ids": list(result.support_address_ids),
        "must_not_store_independent_source_copy": result.must_not_store_independent_source_copy,
    } == expected["expected_output"]
    assert generator.calls == [
        "The ordinary approval limit is ₹25,000 with manager approval."
    ]


def test_paragraph_handoff_is_structurally_no_copy_and_canonical_bound() -> None:
    """Persist references/generated output only and reject a foreign view digest."""
    result, _ = _paragraph_result()
    value = result.to_dict()
    forbidden = {
        "source_text",
        "paragraph_text",
        "evidence_text",
        "excerpt",
        "quoted_source",
        "context_text",
        "canonical_text_copy",
    }
    assert not forbidden.intersection(value)
    assert value["node_spans"] == [{"node_id": "pol-p2"}]

    evidence = frozen_evidence_bundle()
    service, view_set = frozen_paragraph_view()
    foreign = replace(view_set, canonical_content_sha256="0" * 64)
    with pytest.raises(Exception):
        DataForgeHandoffService().build_paragraph_qa(
            evidence=evidence,
            segmentation_service=service,
            view_set=foreign,
            view_id="view-paragraph-v1",
            segment_id="para-2",
            generator=FrozenQuestionAnswerGenerator(),
        )


@pytest.mark.parametrize("status", ["forbidden", "unresolved", "obsolete"])
def test_non_exact_paragraph_support_blocks_generation(status: str) -> None:
    """Reject forbidden, unresolved, and obsolete support before generator use."""
    evidence = frozen_evidence_bundle()
    service, view_set = frozen_paragraph_view()
    generator = FrozenQuestionAnswerGenerator()
    if status == "forbidden":
        resolver = ProvenanceAddressResolver(
            evidence.source_graph,
            evidence.provenance_addresses,
            access_policy=_DenyAll(),
        )
    elif status == "obsolete":
        resolver = ProvenanceAddressResolver(
            evidence.source_graph,
            evidence.provenance_addresses,
            obsolete_address_ids=frozenset({"addr-strong-pol-p2"}),
        )
    else:
        empty = replace(
            evidence.provenance_addresses,
            strong_addresses=(),
            evidence_set_addresses=(),
        )
        resolver = ProvenanceAddressResolver(evidence.source_graph, empty)
    with pytest.raises(HandoffSupportError, match=status):
        DataForgeHandoffService().build_paragraph_qa(
            evidence=evidence,
            segmentation_service=service,
            view_set=view_set,
            view_id="view-paragraph-v1",
            segment_id="para-2",
            generator=generator,
            resolver=resolver,
        )
    assert generator.calls == []


def test_gold_closure_exactly_excludes_ambiguous_candidates() -> None:
    """Follow four graph-ordered gold edges and never traverse candidate targets."""
    evidence = frozen_evidence_bundle()
    closure = build_gold_relation_closure(
        evidence.source_graph, "div-policy-4.2"
    )
    expected = load_fixture("dataforge/composite_ku_contract.json")
    assert list(closure.allowed_relation_ids) == expected["allowed_relation_ids"]
    assert list(closure.excluded_relation_ids) == expected["excluded_relation_ids"]
    assert closure.excluded_relation_ids == ("rel-ambiguous-example",)


def test_composite_ku_matches_frozen_claim_order_and_support() -> None:
    """Generate text only after exact evidence-set closure and preserve all IDs."""
    evidence = frozen_evidence_bundle()
    generator = FrozenClaimGenerator()
    result = DataForgeHandoffService().build_composite_ku(
        evidence=evidence,
        seed_division_id="div-policy-4.2",
        evidence_set_address_id="addr-evidence-ku-travel-approval",
        knowledge_unit_id="ku-travel-approval-001",
        task_schema="travel-expense-approval",
        generator=generator,
    )
    expected = load_fixture("dataforge/composite_ku_contract.json")
    value = result.to_dict()

    assert result.schema == COMPOSITE_KU_HANDOFF_SCHEMA == expected["schema"]
    assert value["allowed_relation_ids"] == expected["allowed_relation_ids"]
    assert value["excluded_relation_ids"] == expected["excluded_relation_ids"]
    for key, expected_value in expected["expected_knowledge_unit"].items():
        assert value[key] == expected_value
    assert [item[0] for item in generator.calls] == [
        "claim-rule",
        "claim-exception",
        "claim-authority",
    ]
    assert [claim.support_address_ids for claim in result.claims] == [
        ("addr-strong-pol-p2",),
        ("addr-strong-pol-p5",),
        ("addr-strong-auth-21",),
    ]


@pytest.mark.parametrize("status", ["forbidden", "unresolved", "obsolete"])
def test_non_exact_evidence_set_blocks_composite_generation(status: str) -> None:
    """Reject incomplete composite support before any claim text is generated."""
    evidence = frozen_evidence_bundle()
    generator = FrozenClaimGenerator()
    if status == "forbidden":
        resolver = ProvenanceAddressResolver(
            evidence.source_graph,
            evidence.provenance_addresses,
            access_policy=_DenyAll(),
        )
    elif status == "obsolete":
        resolver = ProvenanceAddressResolver(
            evidence.source_graph,
            evidence.provenance_addresses,
            obsolete_address_ids=frozenset({"addr-strong-pol-p2"}),
        )
    else:
        empty = replace(
            evidence.provenance_addresses,
            strong_addresses=(),
            evidence_set_addresses=(),
        )
        resolver = ProvenanceAddressResolver(evidence.source_graph, empty)
    with pytest.raises(HandoffSupportError, match="Evidence set"):
        DataForgeHandoffService().build_composite_ku(
            evidence=evidence,
            seed_division_id="div-policy-4.2",
            evidence_set_address_id="addr-evidence-ku-travel-approval",
            knowledge_unit_id="ku-blocked",
            task_schema="fixture",
            generator=generator,
            resolver=resolver,
        )
    assert generator.calls == []


def test_valid_dataforge_owned_evidence_set_intent_is_in_memory_only() -> None:
    """Accept explicit exact intent without mutating the Ingest-owned catalog."""
    evidence = frozen_evidence_bundle()
    catalog = replace(evidence.provenance_addresses, evidence_set_addresses=())
    without_set = ValidatedEvidenceBundle.create(
        evidence.canonical_contents, evidence.source_graph, catalog
    )
    original = load_fixture("dataforge/composite_ku_contract.json")
    intent = EvidenceSetAddress(
        address_id="derived-evidence-set",
        claim_ids=("claim-rule", "claim-exception", "claim-authority"),
        member_address_ids=(
            "addr-strong-pol-p2",
            "addr-strong-pol-p5",
            "addr-strong-auth-21",
        ),
    )
    result = DataForgeHandoffService().build_composite_ku(
        evidence=without_set,
        seed_division_id="div-policy-4.2",
        evidence_set_address_id=intent.address_id,
        evidence_set_intent=intent,
        knowledge_unit_id="ku-travel-approval-001",
        task_schema="travel-expense-approval",
        generator=FrozenClaimGenerator(),
    )
    assert result.evidence_set_address_id == "derived-evidence-set"
    assert catalog.evidence_set_addresses == ()
    assert list(result.allowed_relation_ids) == original["allowed_relation_ids"]


def test_explicit_incomplete_evidence_set_intent_cannot_produce_gold_ku() -> None:
    """Reject DataForge intent whose member is not an existing strong address."""
    evidence = frozen_evidence_bundle()
    catalog = replace(evidence.provenance_addresses, evidence_set_addresses=())
    without_set = ValidatedEvidenceBundle.create(
        evidence.canonical_contents, evidence.source_graph, catalog
    )
    intent = EvidenceSetAddress(
        address_id="derived-set",
        claim_ids=("claim-rule",),
        member_address_ids=("missing-strong-address",),
    )
    with pytest.raises(HandoffValidationError, match="intent"):
        DataForgeHandoffService().build_composite_ku(
            evidence=without_set,
            seed_division_id="div-policy-4.2",
            evidence_set_address_id="derived-set",
            evidence_set_intent=intent,
            knowledge_unit_id="ku-derived",
            task_schema="fixture",
            generator=FrozenClaimGenerator(),
        )


def test_handoff_serialization_and_storage_retry_are_immutable(tmp_path: Path) -> None:
    """Produce stable bytes, accept identical retry, and reject changed identity."""
    result, _ = _paragraph_result()
    assert result.to_json_bytes() == result.to_json_bytes()
    storage = StorageClient(LocalStorageBackend(tmp_path / "storage")).for_shared_data()
    handoffs = HandoffArtifactStore(storage)
    first = handoffs.put("paragraph-proof", result)
    second = handoffs.put("paragraph-proof", result)
    assert first == second
    with pytest.raises(HandoffPersistenceError, match="conflicts"):
        handoffs.put("paragraph-proof", replace(result, answer="changed output"))


def test_t09_adds_no_parser_reparse_vector_semantic_graph_or_recipe_redefinition() -> None:
    """Keep the proof path additive to all three established DataForge recipes."""
    import cognityx_dataforge.handoff as handoff_module

    assert not hasattr(handoff_module, "InferenceClientPool")
    assert not hasattr(handoff_module, "PdfExtractor")
    assert not hasattr(handoff_module, "KnowledgeGraph")
    assert not hasattr(handoff_module, "VectorStore")
    assert SUPPORTED_RECIPES >= {
        "paragraph-qa",
        "knowledge-unit-qa",
        "knowledge-unit-probed-qa",
    }
