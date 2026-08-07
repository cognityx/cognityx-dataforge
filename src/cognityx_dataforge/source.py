"""Resolve Ingest run artifacts without reopening or reparsing source files.

DataForge sits after Ingest. This module turns a logical Storage URI for an
Ingest run or input selection into immutable references that later recipes can
consume. Legacy provenance v1 remains the source of page/block lineage for the
existing recipes. Provenance v2 may additionally advertise canonical content, a
Source Graph, and provenance addresses; together these are the v3.2 handoff.

The design is fail-closed and additive. Direct run-manifest references are a
convenience, not a second authority, so they must exactly match provenance v2.
No function here opens an original PDF, parser-native artifact, local physical
path, model endpoint, vector store, or semantic graph. Storage reads are the only
side effect, and returned records contain URIs and validated metadata rather than
copied source text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from cognityx_storage import StorageClient, StorageRuntime

from cognityx_dataforge.dataset import checksum
from cognityx_dataforge.evidence import load_run_manifest


_PROVENANCE_SCHEMA = "cognityx.ingest.provenance"
_PROVENANCE_V1 = "cognityx.ingest.provenance/v1"
_PROVENANCE_V2 = "cognityx.ingest.provenance/v2"
_MAX_DOCUMENT_ID_LENGTH = 256
_MAX_STORAGE_URI_LENGTH = 4096
_V32_REF_FIELDS = frozenset(
    {
        "document_id",
        "provenance_uri",
        "canonical_content_uri",
        "source_graph_uri",
        "provenance_addresses_uri",
    }
)


class V32HandoffUnavailableError(ValueError):
    """Report that a source is valid for legacy recipes but lacks a T09 bundle.

    ``ResolvedSource.require_v3_2`` constructs this typed failure only when a
    caller selects the new provenance-aware path. Existing recipes never receive
    it merely because they loaded provenance v1. The error is transient, carries
    no source content or physical path, performs no repair, and has ordinary
    immutable exception/thread-safety semantics.
    """


class V32SourceConflictError(V32HandoffUnavailableError):
    """Report contradictory direct and provenance-owned v3.2 artifact URIs.

    Source resolution raises this at the Storage trust boundary instead of
    choosing one conflicting authority. No artifact is loaded, rewritten, or
    merged after a conflict, protecting deterministic support identity for all
    downstream handoff consumers.
    """


@dataclass(frozen=True, slots=True)
class ProvenanceDocument:
    """Retain one validated provenance payload and its logical Storage URI.

    ``_load_provenance`` constructs records in deterministic document-ID order.
    Existing recipes consume the payload's evidence anchors; T09 consumes v2
    ``artifact_uris``. The object owns no storage handle and performs no writes;
    callers must treat the payload mapping as read-only even though JSON mappings
    are not deeply frozen.
    """

    uri: str
    payload: dict[str, Any]

    @property
    def checksum(self) -> str:
        """Return the stable logical checksum used by selection manifests.

        Source-selection recording calls this pure projection. It canonicalizes
        the already-loaded JSON through the existing DataForge checksum helper,
        is deterministic and idempotent, and neither rereads nor persists data.
        """
        return checksum(self.payload)


@dataclass(frozen=True, slots=True)
class ResolvedV32Document:
    """Name the complete T08 artifact set for one successful document.

    ``_resolve_v3_2_bundle`` constructs this record from provenance v2 and,
    optionally, matching run-level direct refs. T09 loaders consume it to locate
    strict canonical, graph, and address artifacts. Every URI must be logical
    Storage identity; source text, physical paths, parser payloads, and mutable
    handles are structurally absent. Frozen scalar fields are safe for concurrent
    reads and deterministic serialization by callers.
    """

    document_id: str
    provenance_uri: str
    canonical_content_uri: str
    source_graph_uri: str
    provenance_addresses_uri: str

    def __post_init__(self) -> None:
        """Validate direct construction at the public source-reference boundary.

        Programmatic callers and ``_parse_v3_2_ref`` converge here so neither can
        create weaker records. The algorithm requires one bounded document ID and
        four nonempty logical ``storage://`` URIs without sorting or dereferencing
        them. It is deterministic, idempotent, side-effect free, and safe for
        concurrent use of frozen values. Invalid input raises
        ``V32SourceConflictError`` before any Storage read or downstream handoff.
        """
        _require_source_text(
            self.document_id,
            "document_id",
            maximum=_MAX_DOCUMENT_ID_LENGTH,
        )
        for name in (
            "provenance_uri",
            "canonical_content_uri",
            "source_graph_uri",
            "provenance_addresses_uri",
        ):
            value = getattr(self, name)
            _require_source_text(value, name, maximum=_MAX_STORAGE_URI_LENGTH)
            if not value.startswith("storage://"):
                raise V32SourceConflictError(
                    f"v3.2 source {name} must be a logical Storage URI"
                )

    def to_dict(self) -> dict[str, str]:
        """Return the closed producer-compatible reference shape.

        Selection-manifest composition and tests call this pure, deterministic
        method. Field order is fixed for readability, no URI is dereferenced, and
        repeated calls return equivalent fresh mappings without side effects.
        """
        return {
            "document_id": self.document_id,
            "provenance_uri": self.provenance_uri,
            "canonical_content_uri": self.canonical_content_uri,
            "source_graph_uri": self.source_graph_uri,
            "provenance_addresses_uri": self.provenance_addresses_uri,
        }


@dataclass(frozen=True, slots=True)
class ResolvedV32SourceBundle:
    """Collect an ordered, complete set of v3.2 document references.

    ``resolve_source`` constructs the bundle only when every provenance document
    in the run is T08-capable. Direct-reference order is retained when supplied;
    fallback order follows deterministic provenance document order. T09 artifact
    loading consumes these immutable records and never merges unrelated graphs or
    reparses source files. The tuple is safe for concurrent readers.
    """

    documents: tuple[ResolvedV32Document, ...]

    def __post_init__(self) -> None:
        """Require immutable records with unique IDs in producer-supplied order.

        ``resolve_source`` and direct callers construct this aggregate. The check
        accepts only a tuple of ``ResolvedV32Document`` values, scans once without
        sorting, and rejects duplicate document IDs. It performs no I/O or
        mutation, preserves deterministic producer order, and raises
        ``V32SourceConflictError`` before loading any artifact. Frozen tuples are
        safe for concurrent readers.
        """
        if not isinstance(self.documents, tuple):
            raise V32SourceConflictError(
                "v3.2 source bundle documents must be an immutable tuple"
            )
        if any(not isinstance(item, ResolvedV32Document) for item in self.documents):
            raise V32SourceConflictError(
                "v3.2 source bundle contains an unsupported document record"
            )
        identities = tuple(item.document_id for item in self.documents)
        if len(identities) != len(set(identities)):
            raise V32SourceConflictError(
                "v3.2 source bundle repeats a document_id"
            )

    def document(self, document_id: str) -> ResolvedV32Document:
        """Return one exact document reference or fail without fuzzy matching.

        T09 composition calls this in-memory lookup after source resolution. It
        preserves stored order, performs no I/O or mutation, and raises
        ``V32HandoffUnavailableError`` when the requested identity is absent.
        """
        _require_source_text(
            document_id,
            "document_id",
            maximum=_MAX_DOCUMENT_ID_LENGTH,
        )
        for item in self.documents:
            if item.document_id == document_id:
                return item
        raise V32HandoffUnavailableError(
            f"v3.2 handoff has no document: {document_id}"
        )


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    """Describe one resolved Ingest selection for old and new recipe paths.

    ``resolve_source`` constructs this aggregate after validating manifests and
    provenance. Existing build code consumes ``provenance`` and selection fields;
    T09 explicitly calls ``require_v3_2``. The record stores no source bytes and
    owns no open handle. Its nested JSON mappings are treated as immutable by
    convention, so concurrent readers must not mutate them.
    """

    submitted_source: str
    source_manifest_uri: str
    source_manifest: dict[str, Any]
    selection_manifest: dict[str, Any]
    provenance: tuple[ProvenanceDocument, ...] = ()
    v3_2: ResolvedV32SourceBundle | None = None

    @property
    def checksum(self) -> str:
        """Return the deterministic checksum of the validated source manifest.

        Build identity and publication lineage call this pure compatibility
        projection. It delegates canonical mapping normalization to the existing
        checksum helper, performs no Storage read or mutation, and returns the
        same value for equivalent manifests. Callers must not mutate the nested
        source mapping concurrently.
        """
        return checksum(self.source_manifest)

    @property
    def evidence_anchor_ids(self) -> dict[str, tuple[str, ...]]:
        """Map evidence records to stable observed anchors from Ingest.

        Existing recipe composition calls this compatibility projection. It scans
        provenance in deterministic document order, preserves first occurrence
        order for page/block anchors, and returns a fresh mapping. It performs no
        I/O, source reconstruction, or mutation; malformed evidence remains the
        responsibility of the established provenance validation path.
        """
        result: dict[str, tuple[str, ...]] = {}
        for document in self.provenance:
            for evidence in document.payload.get("evidence", ()):
                evidence_id = str(evidence.get("evidence_id", ""))
                anchors = tuple(
                    dict.fromkeys(
                        str(value)
                        for value in (
                            evidence.get("anchor_id"),
                            evidence.get("block_id"),
                        )
                        if value
                    )
                )
                if evidence_id:
                    result[evidence_id] = anchors
        return result

    def require_v3_2(self) -> ResolvedV32SourceBundle:
        """Return the complete T09 source bundle or raise a typed failure.

        New provenance-aware composition calls this explicit opt-in boundary.
        Legacy v1 resolution remains successful until this method is requested.
        The lookup is pure, idempotent, and thread-safe for immutable use; it does
        not load parser/native/source artifacts or attempt to repair missing refs.
        """
        if self.v3_2 is None:
            raise V32HandoffUnavailableError(
                "v3.2 handoff unavailable: provenance v2 with canonical, "
                "Source Graph, and provenance-address artifacts is required"
            )
        return self.v3_2


def resolve_storage_uri(
    runtime: StorageRuntime,
    uri: str,
    role_name: str = "artifact",
):
    """Resolve a configured logical Storage URI to a scoped client and key.

    Source and handoff loaders call this boundary with ``storage://`` identities.
    The algorithm selects shared/default or named profiles through
    ``StorageRuntime``, strips only the configured namespace prefix, and returns a
    client plus logical key. It never exposes or accepts a physical root. Backend
    construction may have provider side effects and raises Storage errors;
    malformed/unknown URIs raise ``ValueError``. Thread safety belongs to the
    supplied runtime and backend.
    """
    if not uri.startswith("storage://"):
        raise ValueError(
            "DataForge currently accepts Storage URIs for completed Ingest runs "
            "or DataForge input selections."
        )
    remainder = uri.removeprefix("storage://")
    first, separator, tail = remainder.partition("/")
    if first == "shared":
        profile = runtime.config.default_profile
        if profile is None:
            raise ValueError("Storage configuration has no default profile for shared URI")
        runtime.for_profile(profile, role_name=role_name)
        backend = runtime._backends[profile]  # Runtime owns backend construction.
        return StorageClient(backend).for_shared_data(), tail
    if first not in runtime.config.profiles:
        raise ValueError(f"Unknown storage URI profile: {first}")
    store = runtime.for_profile(first, role_name=role_name)
    key = tail if separator else ""
    namespace = store.namespace.strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    return store, key


def resolve_source(runtime: StorageRuntime, source: str) -> ResolvedSource:
    """Resolve one run/selection while preserving all historical recipe inputs.

    Build orchestration calls this once per submitted logical URI. It reads the
    source JSON, follows an input selection to its run manifest when necessary,
    validates legacy evidence fields, loads provenance v1 or v2, and constructs a
    complete v3.2 bundle only when every document supplies all required T08 refs.
    Run-level direct refs take precedence for ordering but must exactly agree with
    provenance v2. Reads are the only side effects; no parser/source/model is
    invoked. Conflicts fail typed, while absent T08 data remains valid until the
    caller opts into ``require_v3_2``.
    """
    payload = _read_json(runtime, source)
    schema = payload.get("schema") or payload.get("schema_version")
    if schema in {
        "cognityx.dataforge.input-selection/v1",
        "cognityx.dataforge.input-selection",
    }:
        source_manifest_uri = str(payload["source_manifest_uri"])
        source_manifest = load_run_manifest(
            _read_json(runtime, source_manifest_uri)
        )
        selection = dict(payload)
    else:
        source_manifest_uri = source
        source_manifest = load_run_manifest(payload)
        selection = {
            "schema": "cognityx.dataforge.input-selection/v1",
            "submitted_source": source,
            "source_manifest_uri": source_manifest_uri,
            "source_run_id": source_manifest["run_id"],
            "context_id": source_manifest.get("context_id"),
            "source_asset_ids": sorted(
                str(item["asset_id"])
                for item in source_manifest.get("source_assets", ())
                if item.get("asset_id")
            ),
            "document_ids": sorted(
                str(item) for item in source_manifest.get("document_ids", ())
            ),
            "evidence_refs": list(source_manifest["evidence_refs"]),
        }
    provenance = _load_provenance(runtime, source_manifest)
    v3_2 = _resolve_v3_2_bundle(source_manifest, provenance)
    selection["provenance_refs"] = [item.uri for item in provenance]
    selection["provenance_checksums"] = {
        item.uri: item.checksum for item in provenance
    }
    if v3_2 is not None:
        selection["dataforge_source_refs"] = [
            item.to_dict() for item in v3_2.documents
        ]
    return ResolvedSource(
        submitted_source=source,
        source_manifest_uri=source_manifest_uri,
        source_manifest=source_manifest,
        selection_manifest=selection,
        provenance=provenance,
        v3_2=v3_2,
    )


def _read_json(runtime: StorageRuntime, uri: str) -> dict[str, Any]:
    """Read one logical Storage JSON object and require an object root.

    ``resolve_source`` and provenance loading share this bounded helper so every
    read uses the same URI resolver. It opens and closes one Storage handle,
    decodes JSON once, and returns a fresh mapping. The function performs no
    writes or retries; Storage/JSON failures propagate and non-object roots raise
    ``ValueError`` without exposing physical paths.
    """
    store, key = resolve_storage_uri(runtime, uri)
    with store.open(key) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Stored JSON source must contain an object")
    return value


def _load_provenance(
    runtime: StorageRuntime,
    manifest: dict[str, Any],
) -> tuple[ProvenanceDocument, ...]:
    """Load and validate provenance v1/v2 without changing v1 interpretation.

    ``resolve_source`` calls this after run-manifest validation. The algorithm
    reads each declared URI, accepts only the established schema and versions,
    checks unique/expected document and SourceAsset identities plus source hashes,
    then returns records sorted by document ID. It never opens artifact URIs from
    provenance, parser payloads, or source files. Storage reads are the only side
    effect; malformed lineage raises ``ValueError`` and no partial tuple escapes.
    """
    expected_documents = set(manifest.get("document_ids", ()))
    expected_assets = {
        str(item["asset_id"]): item
        for item in manifest.get("source_assets", ())
        if item.get("asset_id")
    }
    documents: list[ProvenanceDocument] = []
    seen: set[str] = set()
    for uri in manifest.get("provenance_refs", ()):
        payload = _read_json(runtime, str(uri))
        if payload.get("schema") != _PROVENANCE_SCHEMA:
            raise ValueError("Unsupported Ingest provenance schema")
        if payload.get("schema_version") not in {_PROVENANCE_V1, _PROVENANCE_V2}:
            raise ValueError("Unsupported Ingest provenance schema version")
        document_id = str(payload.get("document_id", ""))
        if not document_id or document_id in seen:
            raise ValueError("Ingest provenance requires a unique document_id")
        if expected_documents and document_id not in expected_documents:
            raise ValueError("Ingest provenance references an unknown document")
        source_asset = payload.get("source_asset") or {}
        asset_id = str(source_asset.get("asset_id", ""))
        expected_asset = expected_assets.get(asset_id)
        if expected_assets and expected_asset is None:
            raise ValueError("Ingest provenance references an unknown SourceAsset")
        if (
            expected_asset
            and expected_asset.get("sha256")
            and source_asset.get("blob_sha256") != expected_asset["sha256"]
        ):
            raise ValueError("Ingest provenance has inconsistent source content")
        seen.add(document_id)
        documents.append(ProvenanceDocument(str(uri), payload))
    if documents and expected_documents != seen:
        raise ValueError("Ingest provenance does not cover every document")
    return tuple(sorted(documents, key=lambda item: item.payload["document_id"]))


def _resolve_v3_2_bundle(
    manifest: Mapping[str, Any],
    provenance: tuple[ProvenanceDocument, ...],
) -> ResolvedV32SourceBundle | None:
    """Reconcile optional direct refs against authoritative provenance v2.

    ``resolve_source`` calls this pure in-memory algorithm after provenance
    validation. When direct refs are present, it validates their closed shape and
    Storage URIs, preserves run order, and requires field-for-field agreement with
    the matching v2 document. Without direct refs it derives records from complete
    v2 ``artifact_uris`` in deterministic provenance order. Any v1 or incomplete
    v2 run returns ``None`` so legacy recipes remain valid; contradictory asserted
    refs raise ``V32SourceConflictError``. No URI is opened or repaired.
    """
    direct_value = manifest.get("dataforge_source_refs")
    direct_supplied = direct_value is not None
    if direct_supplied and not isinstance(direct_value, list):
        raise V32SourceConflictError("dataforge_source_refs must be an array")
    provenance_by_id = {
        str(item.payload["document_id"]): item for item in provenance
    }
    if direct_supplied:
        records: list[ResolvedV32Document] = []
        seen: set[str] = set()
        for raw in direct_value:
            direct = _parse_v3_2_ref(raw)
            if direct.document_id in seen:
                raise V32SourceConflictError(
                    "dataforge_source_refs repeats a document_id"
                )
            seen.add(direct.document_id)
            document = provenance_by_id.get(direct.document_id)
            if document is None:
                raise V32SourceConflictError(
                    "dataforge_source_refs references unknown provenance"
                )
            derived = _v3_2_ref_from_provenance(document)
            if derived is None or direct != derived:
                raise V32SourceConflictError(
                    "dataforge_source_refs disagree with provenance v2 artifact_uris"
                )
            records.append(direct)
        if set(provenance_by_id) != seen:
            raise V32SourceConflictError(
                "dataforge_source_refs do not cover every provenance document"
            )
        return ResolvedV32SourceBundle(tuple(records)) if records else None

    derived_records: list[ResolvedV32Document] = []
    for document in provenance:
        derived = _v3_2_ref_from_provenance(document)
        if derived is None:
            return None
        derived_records.append(derived)
    return ResolvedV32SourceBundle(tuple(derived_records)) if derived_records else None


def _parse_v3_2_ref(value: object) -> ResolvedV32Document:
    """Parse one closed reference mapping at the run-manifest trust boundary.

    The reconciliation algorithm calls this for each asserted direct ref. It
    rejects missing/extra fields and non-Storage URI values before constructing a
    frozen record. Parsing is deterministic and side-effect free; malformed input
    raises ``V32SourceConflictError`` and never causes URI dereference.
    """
    if not isinstance(value, Mapping) or set(value) != _V32_REF_FIELDS:
        raise V32SourceConflictError(
            "dataforge_source_refs entry has unsupported fields"
        )
    fields = {name: value[name] for name in _V32_REF_FIELDS}
    if any(not isinstance(item, str) for item in fields.values()):
        raise V32SourceConflictError(
            "dataforge_source_refs values must be strings"
        )
    return ResolvedV32Document(**fields)


def _v3_2_ref_from_provenance(
    document: ProvenanceDocument,
) -> ResolvedV32Document | None:
    """Derive one complete ref record from provenance v2 artifact ownership.

    Reconciliation and fallback call this pure helper. Provenance v1 or a v2
    payload missing any required URI returns ``None`` rather than breaking legacy
    resolution. Complete values are validated as logical Storage URIs through the
    same closed parser used for direct refs. No artifact is opened and no value is
    guessed, normalized, or repaired.
    """
    if document.payload.get("schema_version") != _PROVENANCE_V2:
        return None
    artifact_uris = document.payload.get("artifact_uris")
    if not isinstance(artifact_uris, Mapping):
        return None
    required = (
        "canonical_content",
        "source_graph",
        "provenance_addresses",
    )
    if any(not artifact_uris.get(name) for name in required):
        return None
    return _parse_v3_2_ref(
        {
            "document_id": document.payload["document_id"],
            "provenance_uri": document.uri,
            "canonical_content_uri": artifact_uris["canonical_content"],
            "source_graph_uri": artifact_uris["source_graph"],
            "provenance_addresses_uri": artifact_uris["provenance_addresses"],
        }
    )


def _require_source_text(value: object, label: str, *, maximum: int) -> str:
    """Validate one bounded source-reference scalar without coercion or logging.

    Public record construction calls this trust-boundary helper for document IDs
    and logical URIs. It requires an actual nonempty string, strips nothing, and
    returns the original value so identity remains byte-for-byte stable. The pure
    check is deterministic and thread-safe, performs no Storage/network access,
    and raises ``V32SourceConflictError`` without exposing source content or local
    paths.
    """
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise V32SourceConflictError(
            f"v3.2 source {label} must be a bounded nonempty string"
        )
    return value
