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
import hashlib
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
_MAX_TEXT_FIELD_LENGTH = 1_000_000


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

    def __post_init__(self) -> None:
        """Validate direct generated-output construction before service use.

        Generator adapters and deterministic fakes construct this public value.
        Calling ``validate`` here prevents object-like values with a coincidental
        ``strip`` method from crossing the generation trust boundary. The check is
        pure, deterministic, idempotent, and thread-safe for frozen strings; it
        performs no source comparison, I/O, model call, or persistence and raises
        ``HandoffValidationError`` for invalid output.
        """
        self.validate()

    def validate(self) -> None:
        """Require non-empty generated output without comparing it to source text.

        Builders call this pure structural check. Similar wording is legitimate,
        so no substring heuristic is applied. Invalid output raises
        ``HandoffValidationError`` and has no persistence or model side effect.
        """
        _require_handoff_text(
            self.question,
            "question",
            maximum=_MAX_TEXT_FIELD_LENGTH,
        )
        _require_handoff_text(
            self.answer,
            "answer",
            maximum=_MAX_TEXT_FIELD_LENGTH,
        )


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
class _CanonicalEvidenceArtifact:
    """Bind one parsed canonical artifact to its exact deterministic byte digest.

    ``ValidatedEvidenceBundle.create`` constructs this record from the public
    canonical serializer; Storage loaders construct it only after hashing exact
    persisted bytes and proving those bytes equal the serializer output of the
    parsed artifact. Segmentation validation consumes the digest as T06's
    immutable ownership proof. The record stores no extra text or mutable mapping,
    performs no I/O after construction, and is safe for concurrent reads.
    """

    artifact: CanonicalContentArtifact
    canonical_content_sha256: str

    def __post_init__(self) -> None:
        """Prove the digest belongs to the supplied validated canonical artifact.

        Internal constructors and defensive ``dataclasses.replace`` calls converge
        here. The algorithm validates/serializes the artifact, hashes those exact
        deterministic bytes, and compares the supplied lowercase digest. It is
        pure, idempotent, and thread-safe for frozen Ingest records. Mismatch or
        malformed input raises ``HandoffValidationError`` before segmentation,
        resolver, generator, or persistence use.
        """
        if not isinstance(self.artifact, CanonicalContentArtifact):
            raise HandoffValidationError(
                "Canonical evidence record requires CanonicalContentArtifact"
            )
        try:
            payload = self.artifact.to_json_bytes()
        except Exception as error:
            raise HandoffValidationError(
                "Canonical evidence artifact is invalid"
            ) from error
        expected = hashlib.sha256(payload).hexdigest()
        if self.canonical_content_sha256 != expected:
            raise HandoffValidationError(
                "Canonical evidence digest does not match canonical artifact bytes"
            )

    @classmethod
    def from_artifact(
        cls, artifact: CanonicalContentArtifact
    ) -> "_CanonicalEvidenceArtifact":
        """Create a trusted binding from the public deterministic serializer.

        ``ValidatedEvidenceBundle.create`` calls this for in-memory composition.
        It serializes once, computes SHA-256, and delegates defensive proof to the
        frozen constructor. No external digest is accepted, no text is copied into
        metadata, and no Storage/parser/model/network side effect occurs. Equal
        canonical artifacts produce equal records.
        """
        try:
            digest = hashlib.sha256(artifact.to_json_bytes()).hexdigest()
        except Exception as error:
            raise HandoffValidationError(
                "Canonical evidence artifact is invalid"
            ) from error
        return cls(artifact=artifact, canonical_content_sha256=digest)

    @classmethod
    def from_persisted_bytes(
        cls, payload: bytes
    ) -> "_CanonicalEvidenceArtifact":
        """Parse and bind exact persisted canonical bytes at the Storage boundary.

        Per-document and aggregate loaders call this after one configured Storage
        read. The algorithm hashes bytes before parsing, uses strict JSON plus the
        public Ingest reader, and requires canonical reserialization to reproduce
        the exact persisted bytes. This rejects reformatted or mismatched digest
        claims rather than rebinding them. It writes nothing, invokes no parser or
        model, and raises typed validation failures without source excerpts.
        """
        if not isinstance(payload, bytes):
            raise HandoffValidationError("Canonical artifact payload must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        try:
            artifact = CanonicalContentArtifact.from_dict(
                _strict_json_object(payload)
            )
            if artifact.to_json_bytes() != payload:
                raise HandoffValidationError(
                    "Persisted canonical bytes are not the canonical serializer output"
                )
        except HandoffError:
            raise
        except Exception as error:
            raise HandoffValidationError(
                "Canonical content artifact is invalid"
            ) from error
        return cls(artifact=artifact, canonical_content_sha256=digest)


@dataclass(frozen=True, slots=True)
class ValidatedEvidenceBundle:
    """Hold mutually compatible canonical, graph, and address objects by reference.

    ``load`` constructs a shared-graph bundle from resolved T09 Storage URIs,
    ``load_document`` selects one document from a normal run, and ``create``
    serves trusted composition with already-loaded public Ingest types. The main
    algorithm validates each object, requires identical resource-ID/hash coverage,
    proves every strong target belongs to canonical content, and resolves every
    strong address exactly. It never repairs or merges graphs. Multiple canonical
    artifacts are allowed only when one supplied Source Graph already contains
    their explicit cross-resource relations.

    The bundle owns references, not copies or handles. Validation is deterministic
    and idempotent; Storage reads happen only in ``load`` and ``load_document``.
    Nested Ingest records are frozen and safe for concurrent reads.
    """

    _canonical_artifacts: tuple[_CanonicalEvidenceArtifact, ...]
    source_graph: SourceGraph
    provenance_addresses: ProvenanceAddressCatalog

    @property
    def canonical_contents(self) -> tuple[CanonicalContentArtifact, ...]:
        """Return validated canonical objects without exposing mutable digest maps.

        Existing T09 callers and tests use this compatibility projection. It
        preserves supplied artifact order and returns immutable object references;
        digest ownership remains encapsulated in the bundle. The operation is
        deterministic, idempotent, side-effect free, and safe for concurrent
        readers of frozen Ingest records.
        """
        return tuple(item.artifact for item in self._canonical_artifacts)

    @property
    def canonical_content_sha256s(self) -> tuple[str, ...]:
        """Return exact canonical byte digests in canonical artifact order.

        Audit and segmentation tests consume this read-only proof projection. It
        performs no serialization, I/O, rebinding, or mutation and cannot expose
        source text. Equal bundles return equal tuples; construction already
        guarantees every digest belongs to its paired canonical artifact.
        """
        return tuple(
            item.canonical_content_sha256 for item in self._canonical_artifacts
        )

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
        records = tuple(
            _CanonicalEvidenceArtifact.from_artifact(item)
            for item in canonical_contents
        )
        bundle = cls(records, source_graph, provenance_addresses)
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
                "Use load_document(...) for per-document handoff; cross-document "
                "composite requires one pre-existing connected Source Graph and "
                "address catalog, and independent per-document graphs are not merged"
            )
        canonical_artifacts = tuple(
            _load_canonical_record(runtime, item.canonical_content_uri)
            for item in source.documents
        )
        graph = SourceGraph.from_json_bytes(
            _read_storage_bytes(runtime, next(iter(graph_uris)))
        )
        catalog = ProvenanceAddressCatalog.from_json_bytes(
            _read_storage_bytes(runtime, next(iter(address_uris)))
        )
        return cls._from_records(canonical_artifacts, graph, catalog)

    @classmethod
    def load_document(
        cls,
        runtime: StorageRuntime,
        source: ResolvedV32SourceBundle,
        document_id: str,
    ) -> "ValidatedEvidenceBundle":
        """Load one document from a normal multi-document Ingest source bundle.

        Paragraph composition and other per-document T09 consumers call this
        explicit loader. It resolves exactly one validated source-reference record
        without reordering the producer bundle, reads only that document's
        canonical, Source Graph, and provenance-address bytes, hashes canonical
        bytes before strict parsing, and cross-validates the resulting one-document
        evidence bundle. It does not read sibling documents, provenance, parser
        payloads, original sources, models, or networks beyond configured Storage.
        Unknown IDs and incompatible artifacts fail typed; no partial bundle or
        cached handle survives. Thread safety belongs to ``StorageRuntime``.
        """
        document = source.document(document_id)
        canonical = _load_canonical_record(runtime, document.canonical_content_uri)
        graph = SourceGraph.from_json_bytes(
            _read_storage_bytes(runtime, document.source_graph_uri)
        )
        catalog = ProvenanceAddressCatalog.from_json_bytes(
            _read_storage_bytes(runtime, document.provenance_addresses_uri)
        )
        return cls._from_records((canonical,), graph, catalog)

    @classmethod
    def _from_records(
        cls,
        canonical_artifacts: tuple[_CanonicalEvidenceArtifact, ...],
        source_graph: SourceGraph,
        provenance_addresses: ProvenanceAddressCatalog,
    ) -> "ValidatedEvidenceBundle":
        """Finalize already byte-bound canonical records with graph/address facts.

        ``load``, ``load_document``, and trusted creation converge on this internal
        constructor so cross-validation cannot be skipped. It retains deterministic
        canonical order, allocates one frozen bundle, validates once, and returns
        no mutable indexes. The helper performs no I/O, parser/model/network call,
        or persistence; typed handoff failures prevent incompatible facts from
        escaping. Immutable inputs permit concurrent reads after return.
        """
        bundle = cls(canonical_artifacts, source_graph, provenance_addresses)
        bundle.validate()
        return bundle

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
            if (
                not isinstance(self._canonical_artifacts, tuple)
                or not self._canonical_artifacts
            ):
                raise HandoffValidationError("Canonical content collection is empty")
            if any(
                not isinstance(item, _CanonicalEvidenceArtifact)
                for item in self._canonical_artifacts
            ):
                raise HandoffValidationError(
                    "Canonical content collection contains an unbound artifact"
                )
            resources: dict[str, str] = {}
            canonical_ids: set[str] = set()
            for record in self._canonical_artifacts:
                artifact = record.artifact
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
                    "Source Graph resources or source SHA-256 values do not "
                    "match canonical content"
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

    def validate_segmentation_view_set(
        self,
        segmentation_service: SegmentationViewService,
        view_set: SegmentationViewSet,
    ) -> CanonicalContentArtifact:
        """Prove a T06 view set belongs to exactly one canonical artifact here.

        Paragraph handoff calls this production trust method before reconstruction.
        It first delegates full T06 binding/strategy validation to the supplied
        service, then matches the immutable view-set digest to exactly one
        byte-bound canonical record and validates every ordinary, seed, context,
        and retrieval node span against that same artifact's node text length.
        Matching IDs alone are insufficient. The method returns the owning
        canonical artifact by reference, rewrites no digest, persists no text, and
        performs no parser/Storage/model/network call. Foreign bytes or spans fail
        ``HandoffValidationError`` before generator invocation. Frozen records are
        safe for concurrent readers when the segmentation service is.
        """
        self.validate()
        try:
            segmentation_service.validate_view_set(view_set)
        except Exception as error:
            raise HandoffValidationError(
                "Segmentation view set is not valid for its service"
            ) from error
        matches = tuple(
            item
            for item in self._canonical_artifacts
            if item.canonical_content_sha256
            == view_set.canonical_content_sha256
        )
        if len(matches) != 1:
            raise HandoffValidationError(
                "Segmentation canonical digest does not identify exactly one evidence artifact"
            )
        artifact = matches[0].artifact
        nodes = {item.node_id: item for item in artifact.content_nodes}
        for view in view_set.views:
            for segment in view.segments:
                spans = (
                    segment.node_spans
                    + segment.context
                    + segment.retrieval_node_spans
                    + ((segment.seed,) if segment.seed is not None else ())
                )
                for span in spans:
                    node = nodes.get(span.node_id)
                    if node is None:
                        raise HandoffValidationError(
                            "Segmentation span is outside evidence canonical "
                            f"content: {span.node_id}"
                        )
                    try:
                        span.validate(text_length=len(node.content.text))
                    except Exception as error:
                        raise HandoffValidationError(
                            "Segmentation span is invalid for evidence canonical "
                            f"content: {span.node_id}"
                        ) from error
        return artifact

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

    def __post_init__(self) -> None:
        """Validate direct reference construction before any handoff can contain it.

        Service and programmatic callers converge on ``validate`` at construction.
        This makes immutable tuple shape, bounded IDs, span validity, uniqueness,
        and caller-supplied order mandatory before serialization. The operation is
        pure, deterministic, idempotent, side-effect free, and safe for concurrent
        reads; typed failures contain IDs but no reconstructed source text.
        """
        self.validate()

    def validate(self) -> None:
        """Require bounded identities and nonempty unique ordered node spans.

        Direct constructors and ``ParagraphQAHandoff.validate`` call this public
        invariant check. It accepts only a tuple, validates each public ``NodeSpan``
        without resolving text, and rejects duplicate span identities while
        preserving the construction order rather than sorting. Canonical bounds
        are proven later by ``validate_segmentation_view_set``. No I/O, mutation,
        parser, model, or persistence occurs; invalid values fail typed.
        """
        for value, label in (
            (self.source_graph_revision, "source_graph_revision"),
            (self.segmentation_view_id, "segmentation_view_id"),
            (self.segment_id, "segment_id"),
        ):
            _require_handoff_id(value, label)
        if not isinstance(self.node_spans, tuple) or not self.node_spans:
            raise HandoffValidationError(
                "Paragraph handoff node_spans must be a nonempty immutable tuple"
            )
        identities: list[tuple[str, int | None, int | None]] = []
        for span in self.node_spans:
            if not isinstance(span, NodeSpan):
                raise HandoffValidationError(
                    "Paragraph handoff contains an unsupported node span"
                )
            try:
                span.validate()
            except Exception as error:
                raise HandoffValidationError(
                    "Paragraph handoff contains an invalid node span"
                ) from error
            identities.append((span.node_id, span.char_start, span.char_end))
        if len(identities) != len(set(identities)):
            raise HandoffValidationError("Paragraph handoff repeats a node span")

    def to_dict(self) -> dict[str, object]:
        """Return deterministic reference-only JSON data without reconstruction.

        Paragraph handoff serialization calls this pure projection. Graph, view,
        segment, and node-span order are retained exactly; no text, URI, parser
        payload, or physical path can appear. Repeated calls return equivalent
        fresh mappings and raise no new failure after service construction.
        """
        self.validate()
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

    def __post_init__(self) -> None:
        """Validate every directly constructed paragraph handoff immediately.

        The service, future T10 consumers, and tests all receive identical safety
        checks. Delegating to ``validate`` prevents a false no-copy flag or weak
        support tuple from existing as an apparently valid frozen record. The
        operation is deterministic, idempotent, side-effect free, and thread-safe;
        it performs no serialization, I/O, source comparison, or generation.
        """
        self.validate()

    def validate(self) -> None:
        """Enforce the complete paragraph schema and structural no-copy contract.

        Construction and serialization call this public method. It requires the
        exact schema, a valid typed input, real nonempty question/answer strings,
        bounded unique support IDs in supplied order, and boolean ``True`` for the
        no-copy flag. It never compares generated wording with source wording,
        mutates order, or performs I/O. Contradictions raise
        ``HandoffValidationError`` before persistence.
        """
        if self.schema != PARAGRAPH_QA_HANDOFF_SCHEMA:
            raise HandoffValidationError("Unsupported paragraph handoff schema")
        if not isinstance(self.input, ParagraphHandoffInput):
            raise HandoffValidationError(
                "Paragraph handoff requires ParagraphHandoffInput"
            )
        self.input.validate()
        _require_handoff_text(
            self.question, "question", maximum=_MAX_TEXT_FIELD_LENGTH
        )
        _require_handoff_text(
            self.answer, "answer", maximum=_MAX_TEXT_FIELD_LENGTH
        )
        _validate_ordered_ids(self.support_address_ids, "support_address_ids")
        if self.must_not_store_independent_source_copy is not True:
            raise HandoffValidationError(
                "Paragraph handoff no-copy flag must be exactly true"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the closed deterministic paragraph handoff representation.

        Persistence and tests call this pure operation. It validates required
        generated/support values, preserves support order, recursively rejects
        forbidden source-copy field names, and performs no I/O or reconstruction.
        """
        self.validate()
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

    def __post_init__(self) -> None:
        """Validate direct claim construction at the public handoff boundary.

        Composite composition and future deserializers construct this frozen
        value. Calling ``validate`` immediately ensures the generator contributes
        text only while deterministic code contributes bounded, unique support
        IDs. The check is pure, idempotent, side-effect free, and thread-safe; it
        does not compare source wording or perform model/Storage work.
        """
        self.validate()

    def validate(self) -> None:
        """Require one bounded claim ID, generated string, and ordered supports.

        Construction and composite serialization call this public invariant check.
        It retains support order, rejects empty/non-string text and duplicate IDs,
        and introduces no source-context field or heuristic. Invalid state raises
        ``HandoffValidationError`` before persistence; repeated validation is
        deterministic and has no side effects.
        """
        _require_handoff_id(self.claim_id, "claim_id")
        _require_handoff_text(self.text, "claim text", maximum=_MAX_TEXT_FIELD_LENGTH)
        _validate_ordered_ids(self.support_address_ids, "support_address_ids")

    def to_dict(self) -> dict[str, object]:
        """Validate and serialize one claim while preserving support order.

        Composite serialization calls this pure method after generation. It
        requires caller-owned claim identity, non-empty generated text, and at
        least one already-proven support ID; it never reconstructs or stores
        source context. Invalid state raises ``HandoffValidationError`` before
        persistence, and repeated calls are deterministic and side-effect free.
        """
        self.validate()
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

    def __post_init__(self) -> None:
        """Validate direct closure construction before composite publication.

        Gold traversal and programmatic callers converge on ``validate``. The
        check keeps graph order intact while preventing duplicate or contradictory
        allowed/excluded IDs. It is deterministic, idempotent, side-effect free,
        and safe for concurrent frozen reads; no graph traversal, I/O, or source
        reconstruction occurs during validation.
        """
        self.validate()

    def validate(self) -> None:
        """Require a bounded seed and disjoint ordered relation-ID collections.

        Public construction and composite validation call this method. Both
        collections must be immutable tuples of bounded unique IDs; their order is
        accepted as graph-supplied and never sorted. Overlap raises
        ``HandoffValidationError`` because one edge cannot be both gold support and
        excluded audit evidence. The operation performs no external side effect.
        """
        _require_handoff_id(self.seed_division_id, "seed_division_id")
        _validate_relation_partitions(
            self.allowed_relation_ids, self.excluded_relation_ids
        )


@dataclass(frozen=True, slots=True)
class _GoldClosureTraversal:
    """Keep public relation audit data beside internal reachable membership.

    Composite composition uses this private result to prove that every resolved
    evidence member belongs to the bounded traversal rooted at the requested
    division. The public handoff still receives only ``GoldRelationClosure``;
    reachable canonical IDs are an ephemeral validation aid and are never
    serialized as a second source graph or exposed as a new contract.
    """

    closure: GoldRelationClosure
    reachable_canonical_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the immutable traversal proof before composition consumes it.

        The traversal helper is the sole producer. This check requires a valid
        public closure, an ordered unique tuple of bounded canonical IDs, and the
        seed itself in that tuple. It performs no traversal, I/O, or mutation and
        therefore remains deterministic and safe for concurrent frozen reads.
        """
        if not isinstance(self.closure, GoldRelationClosure):
            raise HandoffValidationError("Gold traversal closure is invalid")
        self.closure.validate()
        _validate_ordered_ids(
            self.reachable_canonical_ids, "reachable_canonical_ids"
        )
        if self.closure.seed_division_id not in self.reachable_canonical_ids:
            raise HandoffValidationError(
                "Gold traversal does not retain its seed division"
            )


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

    def __post_init__(self) -> None:
        """Validate direct composite construction before it becomes publishable.

        Service output, programmatic callers, and future T10 readers receive the
        same invariant enforcement. Calling ``validate`` immediately prevents
        contradictory safety booleans, duplicate claims, or overlapping relation
        sets from existing as an apparently valid frozen record. It is pure,
        idempotent, side-effect free, and thread-safe for immutable nested values.
        """
        self.validate()

    def validate(self) -> None:
        """Enforce the complete composite schema, support, and safety contract.

        Construction and serialization call this public method. It requires the
        exact schema, bounded KU/task/evidence-set IDs, a nonempty immutable claim
        tuple with unique claim IDs, valid claims, a valid disjoint relation
        closure shape, and exact booleans: gold-only ``True``, reparse ``False``,
        and no-copy ``True``. It preserves all caller order, persists no text
        context, performs no I/O, and raises ``HandoffValidationError`` on any
        contradiction.
        """
        if self.schema != COMPOSITE_KU_HANDOFF_SCHEMA:
            raise HandoffValidationError("Unsupported composite handoff schema")
        for value, label in (
            (self.knowledge_unit_id, "knowledge_unit_id"),
            (self.task_schema, "task_schema"),
            (self.evidence_set_address_id, "evidence_set_address_id"),
        ):
            _require_handoff_id(value, label)
        if not isinstance(self.claims, tuple) or not self.claims:
            raise HandoffValidationError(
                "Composite handoff claims must be a nonempty immutable tuple"
            )
        if any(not isinstance(item, SupportedClaim) for item in self.claims):
            raise HandoffValidationError(
                "Composite handoff contains an unsupported claim record"
            )
        for item in self.claims:
            item.validate()
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise HandoffValidationError("Composite handoff repeats a claim_id")
        _validate_relation_partitions(
            self.allowed_relation_ids, self.excluded_relation_ids
        )
        if self.gold_support_contains_only_validated_relations is not True:
            raise HandoffValidationError(
                "Composite gold-support flag must be exactly true"
            )
        if self.source_reparse_allowed is not False:
            raise HandoffValidationError(
                "Composite source-reparse flag must be exactly false"
            )
        if self.must_not_store_independent_source_copy is not True:
            raise HandoffValidationError(
                "Composite no-copy flag must be exactly true"
            )

    def to_dict(self) -> dict[str, object]:
        """Return deterministic KU JSON after closed safety validation.

        Persistence and tests call this pure operation. It preserves claim and
        relation order, requires the fixed gold/no-reparse/no-copy booleans, and
        rejects forbidden source-copy field names recursively. It has no external
        side effect and raises ``HandoffValidationError`` before emitting bytes.
        """
        self.validate()
        value = {
            "schema": self.schema,
            "knowledge_unit_id": self.knowledge_unit_id,
            "task_schema": self.task_schema,
            "claims": [item.to_dict() for item in self.claims],
            "evidence_set_address_id": self.evidence_set_address_id,
            "allowed_relation_ids": list(self.allowed_relation_ids),
            "excluded_relation_ids": list(self.excluded_relation_ids),
            "gold_support_contains_only_validated_relations": (
                self.gold_support_contains_only_validated_relations
            ),
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
        evidence.validate_segmentation_view_set(segmentation_service, view_set)
        view = next((item for item in view_set.views if item.view_id == view_id), None)
        if view is None or view.strategy != "paragraph":
            raise HandoffValidationError("Requested paragraph view is unavailable")
        segment = next(
            (item for item in view.segments if item.segment_id == segment_id), None
        )
        if segment is None or not segment.node_spans:
            raise HandoffValidationError("Requested paragraph segment is unavailable")
        support_ids = _support_ids_for_spans(
            evidence.provenance_addresses, segment.node_spans
        )
        selected_resolver = _bind_resolver(
            evidence,
            evidence.provenance_addresses,
            resolver,
        )
        _require_exact_support(selected_resolver, support_ids)
        transient_content = "\n".join(
            segmentation_service.resolve_segment_spans(segment_id, view=view)
        )
        generated = generator.generate_qa(transient_content)
        if not isinstance(generated, GeneratedQuestionAnswer):
            raise HandoffValidationError(
                "Q/A generator returned an unsupported output record"
            )
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
        selected_resolver = _bind_resolver(
            evidence,
            catalog,
            resolver,
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
        traversal = _traverse_gold_relation_closure(
            evidence.source_graph, seed_division_id
        )
        closure = traversal.closure
        reachable = set(traversal.reachable_canonical_ids)
        for member in closure_resolution.member_resolutions:
            if member.target is None or member.target.target_id not in reachable:
                raise HandoffSupportError(
                    "Evidence-set support target is outside the seed's gold closure"
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
            generated_text = generator.generate_claim(claim_id, content)
            if not isinstance(generated_text, str) or not generated_text.strip():
                raise HandoffValidationError(
                    f"Claim generator returned empty text: {claim_id}"
                )
            text = generated_text.strip()
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
        if not isinstance(identity, str) or not _IDENTITY.fullmatch(identity):
            raise HandoffPersistenceError("Handoff identity is invalid")
        if type(handoff) is ParagraphQAHandoff:
            kind = "paragraph-qa"
        elif type(handoff) is CompositeKnowledgeUnitHandoff:
            kind = "composite-ku"
        else:
            raise HandoffPersistenceError(
                "Handoff artifact type is unsupported"
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
    return _traverse_gold_relation_closure(graph, seed_division_id).closure


def _traverse_gold_relation_closure(
    graph: SourceGraph, seed_division_id: str
) -> _GoldClosureTraversal:
    """Traverse gold edges and retain the exact reachable canonical ID set.

    ``build_gold_relation_closure`` exposes the stable relation-only result,
    while composite composition also consumes the reachable IDs returned here.
    The breadth-first walk starts with the seed division, includes each visited
    division's direct canonical nodes, follows only concrete gold-safe relation
    targets, audits non-gold edges without following them, and ignores ambiguous
    candidate targets. Graph order determines both relation and membership order;
    visited IDs bound cycles. No semantic inference, source reconstruction, I/O,
    or mutation occurs.
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
    reachable: set[str] = {seed_division_id}
    reachable_in_order: list[str] = [seed_division_id]
    allowed: set[str] = set()
    excluded: set[str] = set()
    while pending:
        source_id = pending.popleft()
        if source_id in visited:
            continue
        visited.add(source_id)
        division = divisions.get(source_id)
        if division is not None:
            for node_id in division.direct_node_ids:
                if node_id not in reachable:
                    reachable.add(node_id)
                    reachable_in_order.append(node_id)
                if node_id not in visited:
                    pending.append(node_id)
        gold_ids = {
            item.relation_id for item in graph.outgoing(source_id, gold_only=True)
        }
        for relation in graph.outgoing(source_id, gold_only=False):
            if relation.relation_id not in gold_ids:
                excluded.add(relation.relation_id)
                continue
            allowed.add(relation.relation_id)
            if relation.target_id in known_ids:
                if relation.target_id not in reachable:
                    reachable.add(relation.target_id)
                    reachable_in_order.append(relation.target_id)
                if relation.target_id not in visited:
                    pending.append(relation.target_id)
    return _GoldClosureTraversal(
        closure=GoldRelationClosure(
            seed_division_id=seed_division_id,
            allowed_relation_ids=tuple(
                item.relation_id
                for item in graph.relations
                if item.relation_id in allowed
            ),
            excluded_relation_ids=tuple(
                item.relation_id
                for item in graph.relations
                if item.relation_id in excluded
            ),
        ),
        reachable_canonical_ids=tuple(reachable_in_order),
    )


def _load_canonical_record(
    runtime: StorageRuntime, uri: str
) -> _CanonicalEvidenceArtifact:
    """Read, hash, strictly parse, and bind one persisted canonical artifact.

    Aggregate and per-document evidence loaders call this configured Storage
    trust-boundary helper. It reads exact bytes once, delegates hash-before-parse
    and canonical reserialization proof to ``_CanonicalEvidenceArtifact``, and
    returns an immutable digest/artifact pair. It performs no writes, parser,
    original-source, model, or network call beyond the Storage backend. Storage
    and typed handoff failures propagate without payload excerpts or physical
    paths; no cache or open handle survives. Runtime concurrency rules apply.
    """
    return _CanonicalEvidenceArtifact.from_persisted_bytes(
        _read_storage_bytes(runtime, uri)
    )


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


def _bind_resolver(
    evidence: ValidatedEvidenceBundle,
    effective_catalog: ProvenanceAddressCatalog,
    resolver: ProvenanceAddressResolver | None,
) -> ProvenanceAddressResolver:
    """Bind support resolution to byte-equivalent evidence graph/catalog facts.

    Paragraph and composite services call this single trust-boundary helper before
    any support resolution, reconstruction, or generator invocation. With no
    injected resolver it composes the normal public Ingest resolver from the
    evidence graph and operation-effective catalog. An injected resolver may carry
    custom access, version, or obsolescence policy, but its validated graph must
    have the same revision and deterministic bytes as the evidence graph, and its
    effective catalog must serialize byte-for-byte like the ordinary or
    DataForge-intent catalog for this operation. Matching IDs alone never pass.

    The algorithm performs deterministic in-memory validation/serialization only,
    rewrites no resolver or catalog, persists no data, and invokes no policy,
    parser, source, model, network, or Storage operation. Mismatch raises
    ``HandoffValidationError`` before protected target use. Frozen resolver inputs
    are safe to share when their injected policies are thread-safe.
    """
    evidence.validate()
    effective_catalog.validate()
    if resolver is None:
        return ProvenanceAddressResolver(evidence.source_graph, effective_catalog)
    if not isinstance(resolver, ProvenanceAddressResolver):
        raise HandoffValidationError(
            "Injected support resolver has an unsupported type"
        )
    try:
        resolver.graph.validate()
        resolver_catalog = resolver.catalog or resolver.graph.address_catalog
        if resolver_catalog is None:
            raise HandoffValidationError(
                "Injected support resolver has no effective address catalog"
            )
        resolver_catalog.validate()
        if (
            resolver.graph.graph_revision
            != evidence.source_graph.graph_revision
            or resolver.graph.to_json_bytes()
            != evidence.source_graph.to_json_bytes()
        ):
            raise HandoffValidationError(
                "Injected support resolver Source Graph does not match evidence"
            )
        if resolver_catalog.to_json_bytes() != effective_catalog.to_json_bytes():
            raise HandoffValidationError(
                "Injected support resolver address catalog does not match evidence"
            )
    except HandoffError:
        raise
    except Exception as error:
        raise HandoffValidationError(
            "Injected support resolver cannot be validated against evidence"
        ) from error
    return resolver


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


def _require_handoff_text(value: object, label: str, *, maximum: int) -> str:
    """Require a real bounded nonempty string without coercion or source logging.

    Public handoff value validators call this trust-boundary helper for generated
    text and metadata. It rejects object-like values even if they implement
    ``strip``, requires at least one non-whitespace character, preserves the exact
    original string, and bounds memory-facing fields. The pure operation is
    deterministic, idempotent, thread-safe, and performs no I/O, model call,
    source comparison, or persistence. Invalid values raise
    ``HandoffValidationError`` without echoing content.
    """
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
    ):
        raise HandoffValidationError(
            f"Handoff {label} must be a bounded nonempty string"
        )
    return value


def _require_handoff_id(value: object, label: str) -> str:
    """Validate one bounded opaque handoff identity in its original form.

    Public records and traversal helpers call this closed identity check. It
    requires a real string matching the existing 256-character safe identifier
    grammar, returns it unchanged, and never sorts, normalizes, dereferences, or
    logs it. The operation is pure, deterministic, idempotent, side-effect free,
    and safe for concurrent use. Typed validation failure occurs before any
    Storage, resolver, generator, or serialization action.
    """
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise HandoffValidationError(f"Handoff {label} is invalid")
    return value


def _validate_ordered_ids(
    values: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Require an immutable ordered collection of bounded unique identities.

    Claim, support, and relation validators call this shared invariant helper. It
    accepts only a tuple, validates each ID without coercion, rejects duplicates,
    and deliberately retains caller/graph order rather than sorting. Empty tuples
    are accepted only for explicitly optional relation partitions. It performs no
    I/O or mutation, is deterministic and thread-safe, and raises typed validation
    errors before records become publishable.
    """
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise HandoffValidationError(
            f"Handoff {label} must be an immutable ordered tuple"
        )
    for value in values:
        _require_handoff_id(value, label)
    if len(values) != len(set(values)):
        raise HandoffValidationError(f"Handoff {label} contains duplicate IDs")
    return values


def _validate_relation_partitions(
    allowed_relation_ids: object,
    excluded_relation_ids: object,
) -> None:
    """Validate unique ordered and disjoint gold/excluded relation identities.

    ``GoldRelationClosure`` and composite handoff validation share this material
    helper so serialized relation lists obey one rule without inventing a seed
    division. It accepts immutable tuples, allows either partition to be empty,
    validates bounded IDs, and rejects overlap while preserving graph order. The
    algorithm is pure, deterministic, idempotent, thread-safe, and performs no
    graph traversal, I/O, mutation, or persistence. Contradictions raise typed
    handoff validation failures.
    """
    allowed = _validate_ordered_ids(
        allowed_relation_ids,
        "allowed_relation_ids",
        allow_empty=True,
    )
    excluded = _validate_ordered_ids(
        excluded_relation_ids,
        "excluded_relation_ids",
        allow_empty=True,
    )
    overlap = set(allowed).intersection(excluded)
    if overlap:
        raise HandoffValidationError(
            f"Gold closure relation is both allowed and excluded: {sorted(overlap)[0]}"
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
