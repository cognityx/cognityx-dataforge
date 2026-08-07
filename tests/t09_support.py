"""Compose frozen T09 fixture facts through public Ingest production records.

The vendored canonical fixture is intentionally compact, while the persisted T08
canonical schema is complete. Test helpers adapt those unchanged facts into the
public immutable classes, just as the upstream Ingest focused suite does. Expected
Q/A, claims, relation IDs, and support IDs remain in the frozen contract files and
are never derived by production code.
"""

from __future__ import annotations

import json
from pathlib import Path

from cognityx_ingest import (
    CANONICAL_CONTENT_SCHEMA_VERSION,
    CanonicalContentArtifact,
    CanonicalResource,
    CanonicalText,
    ContentNode,
    Division,
    PresentationUnit,
    SegmentationView,
    SegmentationViewService,
    SourceSelector,
)

from cognityx_dataforge.handoff import (
    ClaimTextGenerator,
    GeneratedQuestionAnswer,
    QuestionAnswerGenerator,
    ValidatedEvidenceBundle,
)
from cognityx_ingest import ProvenanceAddressCatalog, SourceGraph


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "t09"


def load_fixture(relative_path: str) -> dict[str, object]:
    """Load one vendored JSON object for deterministic test composition.

    Focused tests call this read-only helper after checksum verification. The
    caller supplies a repository-relative fixture path; the function performs one
    bounded local read and returns the decoded object without changing fixture
    bytes or expected values. Malformed fixture JSON fails the test directly.
    """
    value = json.loads((FIXTURE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def frozen_canonical_artifact() -> CanonicalContentArtifact:
    """Build the complete public canonical type from unchanged compact facts.

    Handoff tests call this deterministic adapter. Resource/node text, hashes,
    selectors, divisions, ownership, and ordering come directly from the vendored
    upstream fixtures. Added production-only metadata is bounded test composition,
    not a replacement oracle. The resulting aggregate validates through Ingest,
    performs no parser/network/Storage/model call, and is safe for shared reads.
    """
    canonical = load_fixture("expected/canonical_content.json")
    graph = load_fixture("expected/source_graph.json")
    resources = tuple(
        sorted(
            (
                CanonicalResource(
                    resource_id=str(item["resource_id"]),
                    source_asset_id=str(item["resource_id"]),
                    source_sha256=str(item["source_sha256"]),
                    media_type="text/markdown",
                    original_filename=Path(str(item["source_path"])).name,
                    logical_uri=f"fixture://{item['source_path']}",
                )
                for item in canonical["resources"]
            ),
            key=lambda item: item.resource_id,
        )
    )
    units = graph["presentation_units"]
    unit_by_resource = {
        str(item["resource_id"]): str(item["presentation_unit_id"])
        for item in units
    }
    presentation_units = tuple(
        PresentationUnit(
            presentation_unit_id=str(item["presentation_unit_id"]),
            resource_id=str(item["resource_id"]),
            unit_type=str(item["unit_type"]),
            sequence_number=index,
        )
        for index, item in enumerate(units)
    )
    node_kind = {
        str(item["node_id"]): str(item["node_kind"])
        for item in canonical["content_nodes"]
    }
    divisions = tuple(
        Division(
            division_id=str(item["division_id"]),
            resource_id=str(item["resource_id"]),
            division_role=str(item["division_role"]),
            parent_division_id=(
                str(item["parent_division_id"])
                if item.get("parent_division_id") is not None
                else None
            ),
            child_division_ids=tuple(str(value) for value in item["child_division_ids"]),
            title_node_id=next(
                (
                    str(node_id)
                    for node_id in item["direct_node_ids"]
                    if node_kind[str(node_id)] == "heading"
                ),
                None,
            ),
            direct_node_ids=tuple(str(value) for value in item["direct_node_ids"]),
            sequence_number=index,
            number=str(item["number"]) if item.get("number") is not None else None,
        )
        for index, item in enumerate(graph["divisions"])
    )
    content_nodes = tuple(
        ContentNode(
            node_id=str(item["node_id"]),
            resource_id=str(item["resource_id"]),
            owner_division_id=str(item["owner_division_id"]),
            node_kind=str(item["node_kind"]),
            content=CanonicalText(
                text=str(item["content"]["text"]),
                sha256=str(item["content"]["sha256"]),
            ),
            source_selectors=tuple(
                SourceSelector(
                    selector_id=f"{item['node_id']}:selector:{selector_index}",
                    selector_type=str(selector["selector_type"]),
                    resource_id=str(item["resource_id"]),
                    presentation_unit_id=unit_by_resource[str(item["resource_id"])],
                    source_path=str(selector["source_path"]),
                    char_start=int(selector["char_start"]),
                    char_end=int(selector["char_end"]),
                )
                for selector_index, selector in enumerate(item["source_selectors"])
            ),
            sequence_number=index,
        )
        for index, item in enumerate(canonical["content_nodes"])
    )
    artifact = CanonicalContentArtifact(
        schema=CANONICAL_CONTENT_SCHEMA_VERSION,
        document_id="fixture-v3-2",
        resources=resources,
        presentation_units=presentation_units,
        divisions=divisions,
        content_nodes=content_nodes,
        representations=(),
        native_bindings=(),
        relations=(),
        processing_activities=(),
        artifact_descriptors=(),
    )
    artifact.validate()
    return artifact


def frozen_evidence_bundle() -> ValidatedEvidenceBundle:
    """Create one validated cross-resource bundle through strict public readers.

    Handoff tests call this after fixture integrity verification. Canonical facts
    use the complete public adapter above; Source Graph and address bytes use the
    strict public compact-fixture readers. Bundle cross-validation then proves
    resource hashes, graph revision, target membership, and exact addresses. No
    parser/source/model/Storage operation occurs.
    """
    graph = SourceGraph.from_json_bytes(
        (FIXTURE_ROOT / "expected/source_graph.json").read_bytes(),
        compact_fixture=True,
    )
    catalog = ProvenanceAddressCatalog.from_json_bytes(
        (FIXTURE_ROOT / "expected/provenance_addresses.json").read_bytes(),
        compact_fixture=True,
    )
    return ValidatedEvidenceBundle.create(
        (frozen_canonical_artifact(),), graph, catalog
    )


def frozen_paragraph_view():
    """Return a canonical-bound service and production paragraph view set.

    Paragraph tests call this adapter because the vendored view file carries the
    upstream compact fixture binding. The algorithm creates the normal service
    from the complete canonical artifact, parses only the frozen paragraph view
    with that exact production digest, and validates the resulting production
    set. It copies references but no source text and performs no external call.
    """
    service = SegmentationViewService.from_canonical(frozen_canonical_artifact())
    fixture = load_fixture("segmentation_views/views.json")
    paragraph = next(
        item for item in fixture["views"] if item["view_id"] == "view-paragraph-v1"
    )
    view = SegmentationView.from_dict(
        paragraph,
        canonical_content_sha256=service.view_set.canonical_content_sha256,
    )
    return service, service.build_view_set((view,))


class FrozenQuestionAnswerGenerator(QuestionAnswerGenerator):
    """Return the exact frozen Q/A while recording only transient call evidence.

    Tests construct this deterministic fake instead of a live model. The contract
    oracle supplies output text; the input content is recorded in memory solely so
    tests can prove reconstruction occurred once. It performs no I/O, persists no
    source context, and is intended for sequential test use.
    """

    def __init__(self) -> None:
        """Load the immutable output oracle and initialize an empty call list."""
        self.expected = load_fixture("dataforge/paragraph_qa_contract.json")[
            "expected_output"
        ]
        self.calls: list[str] = []

    def generate_qa(self, content: str) -> GeneratedQuestionAnswer:
        """Record one transient input and return the frozen generated output."""
        self.calls.append(content)
        return GeneratedQuestionAnswer(
            question=str(self.expected["question"]),
            answer=str(self.expected["answer"]),
        )


class FrozenClaimGenerator(ClaimTextGenerator):
    """Return exact frozen claim text keyed by deterministic caller-owned IDs.

    Composite tests construct this fake to prove support IDs are never generated
    by a model. It records claim IDs and transient content in memory, performs no
    I/O or persistence, and raises ``KeyError`` for an unexpected claim rather
    than inventing output. Sequential test use avoids mutable-list concurrency.
    """

    def __init__(self) -> None:
        """Index frozen claims in authoritative order and start with no calls."""
        expected = load_fixture("dataforge/composite_ku_contract.json")[
            "expected_knowledge_unit"
        ]
        self.text_by_id = {
            str(item["claim_id"]): str(item["text"]) for item in expected["claims"]
        }
        self.calls: list[tuple[str, str]] = []

    def generate_claim(self, claim_id: str, content: str) -> str:
        """Record transient input and return text for the fixed claim identity."""
        self.calls.append((claim_id, content))
        return self.text_by_id[claim_id]
