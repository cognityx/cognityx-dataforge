"""Build provenance-addressed DataForge outputs without copying source content.

This module is the T09 consumer boundary between Cognityx Ingest and DataForge.
Ingest remains the owner of canonical text, segmentation references, Source Graph
facts, and provenance addresses. DataForge reconstructs only the text needed for
one generation call, then persists questions, answers, claims, and support IDs.

The Source Graph is a compact record of explicit source relationships, not a
semantic knowledge graph. Composite traversal follows only graph relations that
Ingest marked as gold-safe; ambiguous candidate targets remain auditable but are
never traversed or accepted as support. All records are immutable and serialize
deterministically. There is no parser execution, source reparse, embedding,
vector database, retrieval ranker, model transport, or CLI in this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import json
import re
from typing import Mapping, Protocol, Sequence

from cognityx_ingest import (
    CanonicalContentArtifact,
    EvidenceSetAddress,
    NodeSpan,
    ProvenanceAddressCatalog,
    ProvenanceAddressResolver,
    SegmentationViewService,
    SegmentationViewSet,
    SourceGraph,
)
from cognityx_storage import ObjectAlreadyExistsError, StorageClient, StorageRuntime

from cognityx_dataforge.inference import GeneratorAdapter, StructuredAdapter
from cognityx_dataforge.source import (
    ResolvedV32SourceBundle,
    resolve_storage_uri,
)


PARAGRAPH_QA_HANDOFF_SCHEMA = "cognityx.dataforge.paragraph-qa-handoff/v1"
COMPOSITE_KU_HANDOFF_SCHEMA = "cognityx.dataforge.composite-ku-handoff/v1"

_FORBIDDEN_COPY_FIELDS = frozenset(
    {
        "source_text",
        "paragraph_text",
        "evidence_text",
        "excerpt",
        "quoted_source",
        "context_text",
        "canonical_text_copy",
    }
)
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class HandoffError(Exception):
    """Base typed failure for the provenance-aware DataForge proof path.

    Public loaders, validators, builders, and persistence adapters construct this
    family so callers do not need to interpret parser, JSON, or implementation
    exceptions. Messages contain IDs and bounded reasons, never reconstructed
    source text, credentials, physical paths, or model prompts. Failures are
    transient and publish no partial artifact.
    """


class HandoffValidationError(HandoffError):
    """Report incompatible canonical, graph, address, or segmentation facts.

    Bundle construction and record serialization raise this error at untrusted
    artifact boundaries. Validation is read-only and fail-closed: mismatched
    hashes, revisions, memberships, or schemas are never repaired or combined.
    """


class HandoffSupportError(HandoffError):
    """Report support that is not exact, complete, and permitted.

    Paragraph and composite builders raise this after using the public Ingest
    resolver. Forbidden, unresolved, obsolete, redirected, or ambiguous results
    cannot produce gold DataForge output. No target details are invented or
    persisted when validation fails.
    """


class HandoffPersistenceError(HandoffError):
    """Report an immutable handoff identity conflict or Storage write failure.

    ``HandoffArtifactStore`` wraps changed-byte retries under one identity with
    this type. The existing object remains untouched. Storage remains the owner
    of physical placement, locking, and provider-level concurrency semantics.
    """


@dataclass(frozen=True, slots=True)
class GeneratedQuestionAnswer:
    """Carry only DataForge-generated question and answer text.

    A narrow Q/A generator constructs this value from transient canonical input.
    ``DataForgeHandoffService`` consumes it immediately while support identity is
    supplied separately by deterministic code. Empty values are rejected; source
    context and model-selected provenance cannot be stored in this closed record.
    Frozen strings are safe for concurrent reads.
    """

    question: str
    answer: str

    def validate(self) -> None:
        """Require non-empty generated output without comparing it to source text.

        Builders call this pure structural check. Similar wording is legitimate,
        so no substring heuristic is applied. Invalid output raises
        ``HandoffValidationError`` and has no persistence or model side effect.
        """
        if not self.question.strip() or not self.answer.strip():
            raise HandoffValidationError("Q/A generator returned empty output")


class QuestionAnswerGenerator(Protocol):
    """Define the narrow generation capability required by paragraph handoff.

    Application composition supplies an adapter around the existing DataForge
    inference boundary; tests supply deterministic fakes. Implementations receive
    transient source content and return generated text only. They must not choose
    support IDs and are responsible for their own client lifecycle/thread safety.
    """

    def generate_qa(self, content: str) -> GeneratedQuestionAnswer:
        """Generate one pair from transient content without choosing support.

        ``DataForgeHandoffService`` calls implementations only after address
        proof. The input must not be retained by the protocol implementation;
        output is generated text only. Implementations own model side effects,
        typed failures, retry semantics, lifecycle, and concurrent-call safety.
        """
        ...


class ClaimTextGenerator(Protocol):
    """Define generation of claim text for an already fixed claim identity.

    Composite composition supplies the claim ID and transient exact support text;
    implementations return text only. They cannot invent evidence-set members,
    relation IDs, or provenance IDs. Client side effects and concurrency behavior
    remain owned by the injected implementation.
    """

    def generate_claim(self, claim_id: str, content: str) -> str:
        """Return text for one fixed claim/support pair without provenance IDs.

        Composite composition calls implementations in evidence-set order after
        exact resolution. The fixed claim ID and transient content are inputs;
        only generated text is output. Implementations own model/network side
        effects, typed failures, retries, lifecycle, and thread safety.
        """
        ...


@dataclass(frozen=True, slots=True)
class GeneratorAdapterQuestionAnswerGenerator:
    """Adapt the existing DataForge ``GeneratorAdapter`` to the narrow Q/A seam.

    Production composition constructs this wrapper around the already configured
    inference adapter. It sends transient content through ``generate``, maps the
    established ``instruction`` output to ``question``, validates both strings,
    and retains no prompt or source context. The wrapped adapter owns network,
    model lifecycle, typed provider failures, idempotency, and thread safety.
    """

    adapter: GeneratorAdapter

    def generate_qa(self, content: str) -> GeneratedQuestionAnswer:
        """Generate and validate Q/A text through the existing adapter.

        ``DataForgeHandoffService`` calls this once after exact support proof. The
        only side effect is the wrapped inference call; support identities never
        cross that boundary. Exceptions propagate from the existing adapter and
        no partial handoff is returned or persisted.
        """
        value = self.adapter.generate(content)
        result = GeneratedQuestionAnswer(
            question=value["instruction"], answer=value["answer"]
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class StructuredAdapterClaimTextGenerator:
    """Adapt the existing ``StructuredAdapter`` to fixed-ID claim generation.

    Application composition constructs this wrapper rather than another generic
    inference client. The algorithm requests strict JSON containing only ``text``
    for a deterministic claim ID and transient support context. It never asks the
    model for addresses or graph relations and retains no context after return.
    The wrapped adapter owns model side effects, retries, and thread safety.
    """

    adapter: StructuredAdapter

    def generate_claim(self, claim_id: str, content: str) -> str:
        """Return one non-empty claim text from the configured inference seam.

        Composite handoff calls this after resolving the member address exactly.
        JSON/model failures propagate or become ``HandoffValidationError`` for an
        empty/invalid shape; no handoff is persisted by this method.
        """
        raw = self.adapter.ask(
            json.dumps({"claim_id": claim_id, "support_context": content}),
            "Return strict JSON with one key named text.",
        )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HandoffValidationError("Claim generator returned invalid JSON") from error
        text = str(value.get("text", "")) if isinstance(value, Mapping) else ""
        if not text.strip():
            raise HandoffValidationError("Claim generator returned empty text")
        return text.strip()


@dataclass(frozen=True, slots=True)
class ValidatedEvidenceBundle:
    """Hold mutually compatible canonical, graph, and address objects by reference.

    ``load`` constructs a bundle from resolved T09 Storage URIs; ``create`` serves
    trusted application/test composition with already-loaded public Ingest types.
    The main algorithm validates each object, requires identical resource-ID/hash
    coverage, proves every strong target belongs to canonical content, and resolves
    every strong address exactly. It never repairs or merges graphs. Multiple
    canonical artifacts are allowed only when one supplied Source Graph already
    contains their explicit cross-resource relations.

    The bundle owns references, not copies or handles. Validation is deterministic
    and idempotent; source/Storage reads happen only in ``load``. Nested Ingest
    records are frozen and safe for concurrent reads.
    """

    canonical_contents: tuple[CanonicalContentArtifact, ...]
    source_graph: SourceGraph
    provenance_addresses: ProvenanceAddressCatalog

    @classmethod
    def create(
        cls,
        canonical_contents: Sequence[CanonicalContentArtifact],
        source_graph: SourceGraph,
        provenance_addresses: ProvenanceAddressCatalog,
    ) -> "ValidatedEvidenceBundle":
        """Validate already-loaded public Ingest objects as one evidence context.

        Services and deterministic tests call this pure constructor. Input order
        is retained, duplicate resource ownership is rejected, and the returned
        tuple is immutable. Typed handoff validation wraps incompatible artifacts;
        no parser, source, network, model, or Storage operation occurs.
        """
        bundle = cls(tuple(canonical_contents), source_graph, provenance_addresses)
        bundle.validate()
        return bundle

    @classmethod
    def load(
        cls,
        runtime: StorageRuntime,
        source: ResolvedV32SourceBundle,
    ) -> "ValidatedEvidenceBundle":
        """Strictly load one complete v3.2 bundle through public Ingest readers.

        T09 application composition calls this after ``resolve_source``. It reads
        every canonical artifact, but requires all documents to name one identical
        Source Graph URI and one identical address-catalog URI before composite
        cross-source work. This deliberately exposes the current limitation: T09
        does not merge unrelated per-document graphs. Only configured Storage is
        read; provenance, parser payloads, original files, models, and local roots
        are untouched. Decode/schema failures become typed validation errors.
        """
        if not source.documents:
            raise HandoffValidationError("v3.2 source bundle is empty")
        graph_uris = {item.source_graph_uri for item in source.documents}
        address_uris = {
            item.provenance_addresses_uri for item in source.documents
        }
        if len(graph_uris) != 1 or len(address_uris) != 1:
            raise HandoffValidationError(
                "Composite handoff requires one pre-existing cross-resource "
                "Source Graph and address catalog; per-document graphs are not merged"
            )
        canonical_contents = tuple(
            _load_canonical(runtime, item.canonical_content_uri)
            for item in source.documents
        )
        graph = SourceGraph.from_json_bytes(
            _read_storage_bytes(runtime, next(iter(graph_uris)))
        )
        catalog = ProvenanceAddressCatalog.from_json_bytes(
            _read_storage_bytes(runtime, next(iter(address_uris)))
        )
        return cls.create(canonical_contents, graph, catalog)

    def validate(self) -> None:
        """Cross-check schemas, revisions, resources, hashes, and target membership.

        Constructors and handoff services call this fail-closed trust-boundary
        algorithm. It validates each Ingest aggregate, builds deterministic local
        indexes, requires exact graph/canonical resource coverage and hashes,
        checks every strong address revision/resource/target, and confirms exact
        default resolver status. It mutates nothing, performs no I/O, is
        idempotent, and wraps incompatible data in ``HandoffValidationError``.
        """
        try:
            if not self.canonical_contents:
                raise HandoffValidationError("Canonical content collection is empty")
            resources: dict[str, str] = {}
            canonical_ids: set[str] = set()
            for artifact in self.canonical_contents:
                artifact.validate()
                for resource in artifact.resources:
                    if resource.resource_id in resources:
                        raise HandoffValidationError(
                            f"Canonical resource is duplicated: {resource.resource_id}"
                        )
                    resources[resource.resource_id] = resource.source_sha256
                canonical_ids.update(item.node_id for item in artifact.content_nodes)
                canonical_ids.update(item.division_id for item in artifact.divisions)
                canonical_ids.update(
                    item.representation_id for item in artifact.representations
                )
            self.source_graph.validate()
            self.provenance_addresses.validate()
            graph_resources = {
                item.resource_id: item.source_sha256
                for item in self.source_graph.resources
            }
            if graph_resources != resources:
                raise HandoffValidationError(
                    "Source Graph resources or source SHA-256 values do not match canonical content"
                )
            resolver = ProvenanceAddressResolver(
                self.source_graph, self.provenance_addresses
            )
            for address in self.provenance_addresses.strong_addresses:
                if address.graph_revision != self.source_graph.graph_revision:
                    raise HandoffValidationError(
                        f"Address graph revision mismatch: {address.address_id}"
                    )
                if address.canonical_target.target_id not in canonical_ids:
                    raise HandoffValidationError(
                        f"Address target is absent from canonical content: {address.address_id}"
                    )
                if (
                    self.source_graph.target_resource_id(address.canonical_target)
                    != address.resource_id
                ):
                    raise HandoffValidationError(
                        f"Address resource does not own target: {address.address_id}"
                    )
                if resolver.resolve(address.address_id).status != "exact":
                    raise HandoffValidationError(
                        f"Strong address is not exact: {address.address_id}"
                    )
        except HandoffError:
            raise
        except Exception as error:
            raise HandoffValidationError(
                "Canonical, Source Graph, and provenance-address artifacts are incompatible"
            ) from error

    def canonical_node_text(self, node_id: str) -> str:
        """Return one transient canonical node value by exact identity.

        Paragraph/composite generation calls this after bundle validation. The
        algorithm scans immutable canonical artifacts, rejects absent/duplicate
        identities, and returns the sole ``CanonicalText.text`` value. It performs
        no I/O, parse, copy persistence, or caching; callers must discard the
        returned string after generation. Typed validation failures contain no
        text. Concurrent reads are safe for immutable inputs.
        """
        matches = [
            node.content.text
            for artifact in self.canonical_contents
            for node in artifact.content_nodes
            if node.node_id == node_id
        ]
        if len(matches) != 1:
            raise HandoffValidationError(
                f"Canonical node does not resolve exactly once: {node_id}"
            )
        return matches[0]

    def canonical_division_text(self, division_id: str) -> str:
        """Reconstruct one division's direct canonical content transiently.

        Composite generation calls this for a division-targeted strong address.
        It finds the sole owning artifact, uses the public ``direct_nodes`` order,
        joins their existing text with newlines for one generator call, and stores
        no derived field or cache. Missing/duplicate divisions fail typed. No
        parser, source file, network, or Storage access occurs.
        """
        owners = [
            artifact
            for artifact in self.canonical_contents
            if any(item.division_id == division_id for item in artifact.divisions)
        ]
        if len(owners) != 1:
            raise HandoffValidationError(
                f"Canonical division does not resolve exactly once: {division_id}"
            )
        return "\n".join(
            item.content.text for item in owners[0].direct_nodes(division_id)
        )


@dataclass(frozen=True, slots=True)
class ParagraphHandoffInput:
    """Retain the exact graph/view/segment references used for paragraph Q/A.

    ``DataForgeHandoffService`` constructs this after canonical-bound view
    validation. It records node/span identities in source order and owns no text.
    The immutable value is serialized as part of the paragraph handoff and can be
    shared safely between readers.
    """

    source_graph_revision: str
    segmentation_view_id: str
    segment_id: str
    node_spans: tuple[NodeSpan, ...]

    def to_dict(self) -> dict[str, object]:
        """Return deterministic reference-only JSON data without reconstruction.

        Paragraph handoff serialization calls this pure projection. Graph, view,
        segment, and node-span order are retained exactly; no text, URI, parser
        payload, or physical path can appear. Repeated calls return equivalent
        fresh mappings and raise no new failure after service construction.
        """
        return {
            "source_graph_revision": self.source_graph_revision,
            "segmentation_view_id": self.segmentation_view_id,
            "segment_id": self.segment_id,
            "node_spans": [item.to_dict() for item in self.node_spans],
        }


@dataclass(frozen=True, slots=True)
class ParagraphQAHandoff:
    """Persist generated paragraph Q/A with exact ordered support addresses.

    The handoff service constructs this immutable output after all support resolves
    exactly and generation succeeds. DataForge dataset preparation and audit are
    primary consumers. Closed fields enforce the no-copy rule structurally; only
    generated question/answer text is present. Serialization is deterministic and
    persistence is delegated to ``HandoffArtifactStore``.
    """

    input: ParagraphHandoffInput
    question: str
    answer: str
    support_address_ids: tuple[str, ...]
    schema: str = PARAGRAPH_QA_HANDOFF_SCHEMA
    must_not_store_independent_source_copy: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return the closed deterministic paragraph handoff representation.

        Persistence and tests call this pure operation. It validates required
        generated/support values, preserves support order, recursively rejects
        forbidden source-copy field names, and performs no I/O or reconstruction.
        """
        if self.schema != PARAGRAPH_QA_HANDOFF_SCHEMA:
            raise HandoffValidationError("Unsupported paragraph handoff schema")
        if not self.question.strip() or not self.answer.strip() or not self.support_address_ids:
            raise HandoffValidationError("Paragraph handoff is incomplete")
        value = {
            "schema": self.schema,
            **self.input.to_dict(),
            "question": self.question,
            "answer": self.answer,
            "support_address_ids": list(self.support_address_ids),
            "must_not_store_independent_source_copy": self.must_not_store_independent_source_copy,
        }
        _reject_copy_fields(value)
        return value

    def to_json_bytes(self) -> bytes:
        """Serialize byte-stable UTF-8 JSON with one trailing newline.

        Storage and reproducibility tests call this idempotent, side-effect-free
        method. Mapping keys are sorted while semantically ordered arrays retain
        their order. Validation occurs before bytes are returned.
        """
        return _canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class SupportedClaim:
    """Pair generated claim text with already-proven support address identities.

    Composite handoff construction creates records in evidence-set member order.
    The generator supplies only ``text``; deterministic code supplies claim and
    support IDs. No reconstructed source context can be represented. The frozen
    value is consumed by KU serialization and audit.
    """

    claim_id: str
    text: str
    support_address_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Validate and serialize one claim while preserving support order.

        Composite serialization calls this pure method after generation. It
        requires caller-owned claim identity, non-empty generated text, and at
        least one already-proven support ID; it never reconstructs or stores
        source context. Invalid state raises ``HandoffValidationError`` before
        persistence, and repeated calls are deterministic and side-effect free.
        """
        if not self.claim_id or not self.text.strip() or not self.support_address_ids:
            raise HandoffValidationError("Supported claim is incomplete")
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "support_address_ids": list(self.support_address_ids),
        }


@dataclass(frozen=True, slots=True)
class GoldRelationClosure:
    """Record deterministic allowed and encountered non-gold Source Graph edges.

    ``build_gold_relation_closure`` constructs this audit result after bounded
    traversal from one seed division. DataForge composite composition consumes it
    as support metadata. Candidate targets are never members, and relation order
    always follows the immutable Source Graph rather than traversal timing.
    """

    seed_division_id: str
    allowed_relation_ids: tuple[str, ...]
    excluded_relation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositeKnowledgeUnitHandoff:
    """Persist one explicit evidence-set Knowledge Unit with gold-safe closure.

    The service constructs this DataForge-owned record from exact Ingest support,
    caller-owned task identity, and generated claim text. It does not overload the
    existing KnowledgeUnit v1 baseline. Dataset/audit consumers receive relation,
    claim, and evidence-set identities but no source context, parser URI, semantic
    expansion, embedding, or reparse permission.
    """

    knowledge_unit_id: str
    task_schema: str
    claims: tuple[SupportedClaim, ...]
    evidence_set_address_id: str
    allowed_relation_ids: tuple[str, ...]
    excluded_relation_ids: tuple[str, ...]
    schema: str = COMPOSITE_KU_HANDOFF_SCHEMA
    gold_support_contains_only_validated_relations: bool = True
    source_reparse_allowed: bool = False
    must_not_store_independent_source_copy: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return deterministic KU JSON after closed safety validation.

        Persistence and tests call this pure operation. It preserves claim and
        relation order, requires the fixed gold/no-reparse/no-copy booleans, and
        rejects forbidden source-copy field names recursively. It has no external
        side effect and raises ``HandoffValidationError`` before emitting bytes.
        """
        if self.schema != COMPOSITE_KU_HANDOFF_SCHEMA:
            raise HandoffValidationError("Unsupported composite handoff schema")
        if (
            not self.knowledge_unit_id
            or not self.task_schema
            or not self.claims
            or not self.evidence_set_address_id
            or not self.gold_support_contains_only_validated_relations
            or self.source_reparse_allowed
            or not self.must_not_store_independent_source_copy
        ):
            raise HandoffValidationError("Composite handoff violates T09 invariants")
        value = {
            "schema": self.schema,
            "knowledge_unit_id": self.knowledge_unit_id,
            "task_schema": self.task_schema,
            "claims": [item.to_dict() for item in self.claims],
            "evidence_set_address_id": self.evidence_set_address_id,
            "allowed_relation_ids": list(self.allowed_relation_ids),
            "excluded_relation_ids": list(self.excluded_relation_ids),
            "gold_support_contains_only_validated_relations": self.gold_support_contains_only_validated_relations,
            "source_reparse_allowed": self.source_reparse_allowed,
            "must_not_store_independent_source_copy": self.must_not_store_independent_source_copy,
        }
        _reject_copy_fields(value)
        return value

    def to_json_bytes(self) -> bytes:
        """Serialize byte-stable UTF-8 JSON after complete KU validation.

        ``HandoffArtifactStore`` and reproducibility tests call this pure method.
        It delegates closed-field validation to ``to_dict``, preserves claim and
        relation order, sorts mapping keys, and returns identical bytes for equal
        records. It performs no Storage, parser, source, model, or network action.
        """
        return _canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class DataForgeHandoffService:
    """Compose paragraph and composite outputs from exact Ingest support.

    Application code and tests construct this stateless service. Its algorithms
    validate the supplied evidence bundle and segmentation bindings, reconstruct
    bounded text only for injected generators, resolve every support address, and
    return immutable DataForge-owned outputs. It owns no Storage client, parser,
    inference transport, cache, or persistent state. Calls are deterministic when
    generators are deterministic; concurrent use requires thread-safe injected
    generators and resolvers.
    """

    def build_paragraph_qa(
        self,
        *,
        evidence: ValidatedEvidenceBundle,
        segmentation_service: SegmentationViewService,
        view_set: SegmentationViewSet,
        view_id: str,
        segment_id: str,
        generator: QuestionAnswerGenerator,
        resolver: ProvenanceAddressResolver | None = None,
    ) -> ParagraphQAHandoff:
        """Build one Q/A handoff from a canonical-bound paragraph segment.

        T09 paragraph composition calls this with validated Ingest objects and a
        narrow generator. The algorithm validates the bundle and production view
        set, locates one paragraph segment, reconstructs its ordered spans only in
        memory, finds strong addresses with exactly matching node/range targets,
        requires exact resolver status, invokes generation once, and returns a
        no-copy record. It performs no parser/source/Storage operation and retains
        no transient context. Typed validation/support failures prevent output;
        generator side effects and thread safety belong to the injected object.
        """
        evidence.validate()
        segmentation_service.validate_view_set(view_set)
        view = next((item for item in view_set.views if item.view_id == view_id), None)
        if view is None or view.strategy != "paragraph":
            raise HandoffValidationError("Requested paragraph view is unavailable")
        segment = next(
            (item for item in view.segments if item.segment_id == segment_id), None
        )
        if segment is None or not segment.node_spans:
            raise HandoffValidationError("Requested paragraph segment is unavailable")
        transient_content = "\n".join(
            segmentation_service.resolve_segment_spans(
                segment_id, view=view
            )
        )
        support_ids = _support_ids_for_spans(
            evidence.provenance_addresses, segment.node_spans
        )
        selected_resolver = resolver or ProvenanceAddressResolver(
            evidence.source_graph, evidence.provenance_addresses
        )
        _require_exact_support(selected_resolver, support_ids)
        generated = generator.generate_qa(transient_content)
        generated.validate()
        return ParagraphQAHandoff(
            input=ParagraphHandoffInput(
                source_graph_revision=evidence.source_graph.graph_revision,
                segmentation_view_id=view_id,
                segment_id=segment_id,
                node_spans=segment.node_spans,
            ),
            question=generated.question.strip(),
            answer=generated.answer.strip(),
            support_address_ids=support_ids,
        )

    def build_composite_ku(
        self,
        *,
        evidence: ValidatedEvidenceBundle,
        seed_division_id: str,
        evidence_set_address_id: str,
        knowledge_unit_id: str,
        task_schema: str,
        generator: ClaimTextGenerator,
        evidence_set_intent: EvidenceSetAddress | None = None,
        resolver: ProvenanceAddressResolver | None = None,
    ) -> CompositeKnowledgeUnitHandoff:
        """Build one explicit evidence-set KU using only gold Source Graph closure.

        Deterministic task composition supplies all IDs; the generator supplies
        claim text only. The algorithm validates the evidence bundle, optionally
        adds a DataForge-owned evidence-set intent to an in-memory catalog copy,
        requires complete exact evidence-set/member resolution, computes bounded
        gold-only relation closure, reconstructs each node/division target only for
        its generation call, and preserves claim/member order. No Ingest artifact
        is mutated, no candidate target is traversed, and no parser/source/Storage
        operation occurs. Typed support/validation failures publish nothing.
        """
        evidence.validate()
        catalog = _catalog_with_intent(
            evidence.provenance_addresses,
            evidence_set_address_id,
            evidence_set_intent,
        )
        selected_resolver = resolver or ProvenanceAddressResolver(
            evidence.source_graph, catalog
        )
        evidence_set = catalog.get(evidence_set_address_id)
        if not isinstance(evidence_set, EvidenceSetAddress):
            raise HandoffValidationError(
                "Composite handoff requires an evidence-set address"
            )
        closure_resolution = selected_resolver.resolve(evidence_set_address_id)
        if (
            closure_resolution.status != "exact"
            or len(closure_resolution.member_resolutions)
            != len(evidence_set.member_address_ids)
            or any(
                item.status != "exact"
                for item in closure_resolution.member_resolutions
            )
        ):
            raise HandoffSupportError(
                f"Evidence set is not complete and exact: {evidence_set_address_id}"
            )
        closure = build_gold_relation_closure(
            evidence.source_graph, seed_division_id
        )
        claims: list[SupportedClaim] = []
        for claim_id, address_id in zip(
            evidence_set.claim_ids,
            evidence_set.member_address_ids,
            strict=True,
        ):
            resolution = selected_resolver.resolve(address_id)
            if resolution.status != "exact" or resolution.target is None:
                raise HandoffSupportError(
                    f"Claim support is not exact: {address_id} ({resolution.status})"
                )
            content = _target_content(evidence, resolution.target)
            text = generator.generate_claim(claim_id, content).strip()
            if not text:
                raise HandoffValidationError(
                    f"Claim generator returned empty text: {claim_id}"
                )
            claims.append(
                SupportedClaim(
                    claim_id=claim_id,
                    text=text,
                    support_address_ids=(address_id,),
                )
            )
        return CompositeKnowledgeUnitHandoff(
            knowledge_unit_id=knowledge_unit_id,
            task_schema=task_schema,
            claims=tuple(claims),
            evidence_set_address_id=evidence_set_address_id,
            allowed_relation_ids=closure.allowed_relation_ids,
            excluded_relation_ids=closure.excluded_relation_ids,
        )


@dataclass(frozen=True, slots=True)
class HandoffArtifactStore:
    """Persist immutable DataForge handoffs through a configured Storage role.

    Application composition constructs this seam with the DataForge dataset or
    artifact ``StorageClient`` obtained from ``StorageRuntime``. ``put`` derives a
    logical key from schema and caller identity, writes deterministic bytes, treats
    an identical existing object as an idempotent retry, and rejects changed bytes
    instead of overwriting. No physical root or path enters the API. Backend
    locking/thread safety and provider lifecycle remain Storage responsibilities.
    """

    storage: StorageClient

    def put(
        self,
        identity: str,
        handoff: ParagraphQAHandoff | CompositeKnowledgeUnitHandoff,
    ) -> str:
        """Write one deterministic artifact and return its logical Storage URI.

        DataForge publication calls this after handoff construction. The method
        validates a bounded identity, serializes once, and uses immutable
        ``put_bytes``. On an existing key it compares exact bytes: identical retry
        returns the existing URI; conflict raises ``HandoffPersistenceError`` and
        leaves stored data unchanged. Storage failures are wrapped without local
        paths or payload content. No parser/model/source side effect occurs.
        """
        if not _IDENTITY.fullmatch(identity):
            raise HandoffPersistenceError("Handoff identity is invalid")
        kind = (
            "paragraph-qa"
            if isinstance(handoff, ParagraphQAHandoff)
            else "composite-ku"
        )
        key = f"dataforge/handoffs/{kind}/{identity}.json"
        payload = handoff.to_json_bytes()
        try:
            stored = self.storage.put_bytes(
                key, payload, media_type="application/json"
            )
        except ObjectAlreadyExistsError:
            with self.storage.open(key) as existing:
                if existing.read() != payload:
                    raise HandoffPersistenceError(
                        f"Immutable handoff conflicts with existing identity: {identity}"
                    ) from None
            stored = self.storage.stat(key)
        except Exception as error:
            raise HandoffPersistenceError(
                f"Unable to persist handoff identity: {identity}"
            ) from error
        return str(stored.uri)


def build_gold_relation_closure(
    graph: SourceGraph, seed_division_id: str
) -> GoldRelationClosure:
    """Traverse only explicit gold-safe Source Graph relations from one division.

    Composite handoff calls this bounded in-memory algorithm. It begins with the
    seed and each encountered division's direct nodes, inspects all outgoing edges
    for audit, follows only ``graph.outgoing(..., gold_only=True)`` concrete
    targets, records back-edges even to visited targets, and never traverses
    ``candidate_target_ids``. Visited IDs prevent cycles. After membership is
    known, allowed and excluded IDs are emitted in immutable graph relation order,
    independent of queue timing. No semantic inference, graph database, I/O, or
    mutation occurs; invalid seeds raise ``HandoffValidationError``.
    """
    graph.validate()
    divisions = {item.division_id: item for item in graph.divisions}
    known_ids = set(divisions)
    known_ids.update(
        node_id for division in graph.divisions for node_id in division.direct_node_ids
    )
    known_ids.update(item.representation_id for item in graph.representations)
    if seed_division_id not in divisions:
        raise HandoffValidationError(
            f"Composite seed division is unavailable: {seed_division_id}"
        )
    pending: deque[str] = deque((seed_division_id,))
    visited: set[str] = set()
    allowed: set[str] = set()
    excluded: set[str] = set()
    while pending:
        source_id = pending.popleft()
        if source_id in visited:
            continue
        visited.add(source_id)
        division = divisions.get(source_id)
        if division is not None:
            pending.extend(
                node_id
                for node_id in division.direct_node_ids
                if node_id not in visited
            )
        gold_ids = {
            item.relation_id for item in graph.outgoing(source_id, gold_only=True)
        }
        for relation in graph.outgoing(source_id, gold_only=False):
            if relation.relation_id not in gold_ids:
                excluded.add(relation.relation_id)
                continue
            allowed.add(relation.relation_id)
            if relation.target_id in known_ids and relation.target_id not in visited:
                pending.append(relation.target_id)
    return GoldRelationClosure(
        seed_division_id=seed_division_id,
        allowed_relation_ids=tuple(
            item.relation_id for item in graph.relations if item.relation_id in allowed
        ),
        excluded_relation_ids=tuple(
            item.relation_id
            for item in graph.relations
            if item.relation_id in excluded
        ),
    )


def _load_canonical(
    runtime: StorageRuntime, uri: str
) -> CanonicalContentArtifact:
    """Strictly load one production canonical artifact from configured Storage.

    ``ValidatedEvidenceBundle.load`` calls this at the persisted trust boundary.
    It reads bytes once, rejects duplicate JSON keys/non-object roots, delegates
    complete shape and invariant validation to the public Ingest
    ``CanonicalContentArtifact.from_dict``, and returns the frozen aggregate. It
    performs no writes, parser/source/model access, or repair; failures are wrapped
    in ``HandoffValidationError`` without exposing payload text.
    """
    try:
        value = _strict_json_object(_read_storage_bytes(runtime, uri))
        return CanonicalContentArtifact.from_dict(value)
    except HandoffError:
        raise
    except Exception as error:
        raise HandoffValidationError("Canonical content artifact is invalid") from error


def _read_storage_bytes(runtime: StorageRuntime, uri: str) -> bytes:
    """Read and close one logical artifact without resolving a physical path.

    Bundle loading calls this bounded helper for canonical, graph, and address
    bytes. Configured Storage reads are its only side effect; it does not cache,
    parse, reparse, or mutate data. Storage/URI failures propagate to the caller's
    typed validation boundary. Thread safety belongs to the supplied runtime.
    """
    store, key = resolve_storage_uri(runtime, uri)
    with store.open(key) as handle:
        return handle.read()


def _strict_json_object(payload: bytes) -> dict[str, object]:
    """Decode UTF-8 JSON while rejecting duplicate keys and non-object roots.

    The canonical reader adapter calls this before public Ingest deserialization.
    It is deterministic, side-effect free, and accepts no non-standard constants.
    Malformed input raises ``HandoffValidationError`` with no payload excerpt.
    """
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        """Build one mapping while rejecting duplicate names at every depth.

        Python's JSON decoder calls this local trust-boundary hook for each object.
        It preserves encounter order, allocates one fresh mapping, and raises a
        typed failure before later keys can hide earlier values. It performs no
        I/O or mutation outside that local mapping and retains no payload bytes.
        """
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise HandoffValidationError("Stored JSON contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                HandoffValidationError("Stored JSON contains a non-finite number")
            ),
        )
    except HandoffError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise HandoffValidationError("Stored artifact is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise HandoffValidationError("Stored artifact must contain a JSON object")
    return value


def _support_ids_for_spans(
    catalog: ProvenanceAddressCatalog, spans: tuple[NodeSpan, ...]
) -> tuple[str, ...]:
    """Find exact whole/ranged node addresses in segment and catalog order.

    Paragraph composition calls this after canonical-bound span validation. For
    each segment span in order, it selects strong addresses whose node target and
    optional half-open range match exactly, preserving catalog order and removing
    duplicate IDs. Every span must have support. No resolver, I/O, source content,
    or heuristic selector matching is used; absence raises typed support failure.
    """
    selected: list[str] = []
    for span in spans:
        matches = [
            address.address_id
            for address in catalog.strong_addresses
            if address.canonical_target.node_id == span.node_id
            and address.canonical_target.char_start == span.char_start
            and address.canonical_target.char_end == span.char_end
        ]
        if not matches:
            raise HandoffSupportError(
                f"No exact strong address covers segment node span: {span.node_id}"
            )
        selected.extend(item for item in matches if item not in selected)
    return tuple(selected)


def _require_exact_support(
    resolver: ProvenanceAddressResolver, address_ids: Sequence[str]
) -> None:
    """Require every ordered address to resolve with status exactly ``exact``.

    Paragraph support validation calls this pure resolver loop. It preserves input
    order, accepts no redirect/ambiguity/absence/obsolescence/forbidden result, and
    raises ``HandoffSupportError`` before generation. Resolver policies may have
    their own side effects and thread-safety contract; no source text is exposed.
    """
    for address_id in address_ids:
        resolution = resolver.resolve(address_id)
        if resolution.status != "exact":
            raise HandoffSupportError(
                f"Support is not exact: {address_id} ({resolution.status})"
            )


def _catalog_with_intent(
    catalog: ProvenanceAddressCatalog,
    address_id: str,
    intent: EvidenceSetAddress | None,
) -> ProvenanceAddressCatalog:
    """Return an immutable catalog view containing explicit DataForge intent.

    Composite composition calls this before resolution. Existing evidence-set
    identity remains authoritative; an equal explicit value is accepted, a
    conflict fails, and a new explicit intent is appended to an in-memory catalog
    copy then validated against existing strong members. The original Ingest
    artifact is never mutated or rewritten. The operation is pure, deterministic,
    and raises typed validation errors for incomplete or conflicting intent.
    """
    if intent is not None and intent.address_id != address_id:
        raise HandoffValidationError("Evidence-set intent identity does not match request")
    existing = catalog.find(address_id)
    if existing is not None:
        if not isinstance(existing, EvidenceSetAddress):
            raise HandoffValidationError("Requested evidence-set ID has another address type")
        if intent is not None and intent != existing:
            raise HandoffValidationError("Explicit evidence-set intent conflicts with catalog")
        return catalog
    if intent is None:
        raise HandoffSupportError(f"Evidence-set address is unavailable: {address_id}")
    derived = replace(
        catalog,
        evidence_set_addresses=(*catalog.evidence_set_addresses, intent),
    )
    try:
        derived.validate()
    except Exception as error:
        raise HandoffValidationError("Explicit evidence-set intent is invalid") from error
    return derived


def _target_content(evidence: ValidatedEvidenceBundle, target: object) -> str:
    """Reconstruct one exact resolved node/span or division target transiently.

    Composite generation calls this only with a public ``ProvenanceTarget`` from
    an exact resolver result. Node targets use canonical text and optional
    half-open ranges; division targets use deterministic direct canonical content.
    Representations are deliberately unsupported because T09 defines no safe text
    reconstruction for them. No value is cached or persisted, and unsupported or
    invalid ranges raise ``HandoffValidationError`` without exposing text.
    """
    node_id = getattr(target, "node_id", None)
    division_id = getattr(target, "division_id", None)
    if node_id is not None:
        content = evidence.canonical_node_text(node_id)
        start = getattr(target, "char_start", None)
        end = getattr(target, "char_end", None)
        if start is None:
            return content
        if end is None or end > len(content):
            raise HandoffValidationError("Resolved node target range is invalid")
        return content[start:end]
    if division_id is not None:
        return evidence.canonical_division_text(division_id)
    raise HandoffValidationError(
        "T09 claim generation supports canonical node or division targets only"
    )


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    """Encode deterministic compact UTF-8 JSON with one trailing newline.

    Both handoff record types call this pure serializer after validation. It sorts
    object keys, preserves ordered arrays and Unicode, rejects unsupported values,
    and returns identical bytes for identical records. No I/O occurs; encoding
    failures become ``HandoffValidationError``.
    """
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HandoffValidationError("Handoff contains non-JSON data") from error


def _reject_copy_fields(value: object) -> None:
    """Enforce the no-copy rule by closed field names, never text similarity.

    Handoff serializers call this recursive structural check over their own small
    JSON trees. It rejects only explicitly prohibited independent source-context
    fields, so legitimate generated answers/claims may quote source wording. The
    walk is deterministic, side-effect free, and raises typed validation failure;
    it does not inspect values, call models, or persist data.
    """
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_COPY_FIELDS.intersection(value)
        if forbidden:
            raise HandoffValidationError(
                f"Handoff contains forbidden source-copy field: {sorted(forbidden)[0]}"
            )
        for item in value.values():
            _reject_copy_fields(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_copy_fields(item)
