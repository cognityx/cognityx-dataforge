from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cognityx_storage import StorageClient, StorageRuntime

from cognityx_dataforge.dataset import checksum
from cognityx_dataforge.evidence import load_run_manifest


@dataclass(frozen=True, slots=True)
class ProvenanceDocument:
    uri: str
    payload: dict[str, Any]

    @property
    def checksum(self) -> str:
        return checksum(self.payload)


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    submitted_source: str
    source_manifest_uri: str
    source_manifest: dict[str, Any]
    selection_manifest: dict[str, Any]
    provenance: tuple[ProvenanceDocument, ...] = ()

    @property
    def checksum(self) -> str:
        return checksum(self.source_manifest)

    @property
    def evidence_anchor_ids(self) -> dict[str, tuple[str, ...]]:
        """Map evidence records to stable observed anchors from Ingest."""
        result: dict[str, tuple[str, ...]] = {}
        for document in self.provenance:
            for evidence in document.payload.get("evidence", ()):
                evidence_id = str(evidence.get("evidence_id", ""))
                anchors = tuple(dict.fromkeys(
                    str(value)
                    for value in (
                        evidence.get("anchor_id"),
                        evidence.get("block_id"),
                    )
                    if value
                ))
                if evidence_id:
                    result[evidence_id] = anchors
        return result


def resolve_storage_uri(
    runtime: StorageRuntime,
    uri: str,
    role_name: str = "artifact",
):
    """Resolve a configured Storage URI without exposing its physical backend."""
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
    store, key = resolve_storage_uri(runtime, source)
    with store.open(key) as handle:
        payload = json.load(handle)

    schema = payload.get("schema") or payload.get("schema_version")
    if schema in {
        "cognityx.dataforge.input-selection/v1",
        "cognityx.dataforge.input-selection",
    }:
        source_manifest_uri = str(payload["source_manifest_uri"])
        manifest_store, manifest_key = resolve_storage_uri(runtime, source_manifest_uri)
        with manifest_store.open(manifest_key) as handle:
            source_manifest = load_run_manifest(json.load(handle))
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
    selection["provenance_refs"] = [item.uri for item in provenance]
    selection["provenance_checksums"] = {
        item.uri: item.checksum for item in provenance
    }
    return ResolvedSource(
        submitted_source=source,
        source_manifest_uri=source_manifest_uri,
        source_manifest=source_manifest,
        selection_manifest=selection,
        provenance=provenance,
    )


def _load_provenance(
    runtime: StorageRuntime,
    manifest: dict[str, Any],
) -> tuple[ProvenanceDocument, ...]:
    expected_documents = set(manifest.get("document_ids", ()))
    expected_assets = {
        str(item["asset_id"]): item
        for item in manifest.get("source_assets", ())
        if item.get("asset_id")
    }
    documents: list[ProvenanceDocument] = []
    seen: set[str] = set()
    for uri in manifest.get("provenance_refs", ()):
        store, key = resolve_storage_uri(runtime, str(uri))
        with store.open(key) as handle:
            payload = json.load(handle)
        if payload.get("schema") != "cognityx.ingest.provenance":
            raise ValueError("Unsupported Ingest provenance schema")
        if payload.get("schema_version") != "cognityx.ingest.provenance/v1":
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
