"""Exercise provenance v1/v2 and direct-reference source resolution for T09."""

from __future__ import annotations

from pathlib import Path

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.source import (
    V32HandoffUnavailableError,
    V32SourceConflictError,
    resolve_source,
)


def _stored_run(
    tmp_path: Path,
    *,
    version: str,
    direct_refs: bool = False,
    conflict: bool = False,
):
    """Persist one minimal Ingest run through normal configured Storage.

    Focused source tests call this helper with v1 or v2. It writes only the
    provenance and run JSON needed by ``resolve_source``; all advertised T08 and
    parser URIs are deliberately absent objects, proving source resolution does
    not load them. The temporary runtime owns lifecycle and is not shared.
    """
    runtime = StorageRuntime.from_config(
        StorageConfig.built_in(root=tmp_path / "storage")
    )
    artifact = runtime.for_role("artifact")
    artifact_uris = {
        "canonical_content": "storage://shared/ingest/doc/canonical-content.json",
        "source_graph": "storage://shared/ingest/doc/source-graph.json",
        "provenance_addresses": "storage://shared/ingest/doc/provenance-addresses.json",
        "parser": {"never-open": "storage://shared/ingest/doc/parser/raw.json"},
    }
    provenance_uri = artifact.put_json(
        "ingest/doc/provenance.json",
        {
            "schema": "cognityx.ingest.provenance",
            "schema_version": version,
            "document_id": "document-1",
            "source_asset": {
                "asset_id": "asset-1",
                "blob_sha256": "source-sha",
            },
            **({"artifact_uris": artifact_uris} if version.endswith("/v2") else {}),
            "evidence": [],
        },
    ).uri
    refs = [
        {
            "document_id": "document-1",
            "provenance_uri": provenance_uri,
            "canonical_content_uri": (
                "storage://shared/wrong.json"
                if conflict
                else artifact_uris["canonical_content"]
            ),
            "source_graph_uri": artifact_uris["source_graph"],
            "provenance_addresses_uri": artifact_uris["provenance_addresses"],
        }
    ]
    manifest = {
        "schema": "cognityx.ingest.run",
        "run_id": "run-1",
        "context_id": "context-1",
        "source_assets": [{"asset_id": "asset-1", "sha256": "source-sha"}],
        "document_ids": ["document-1"],
        "evidence_refs": [],
        "provenance_refs": [provenance_uri],
    }
    if direct_refs:
        manifest["dataforge_source_refs"] = refs
    manifest_uri = artifact.put_json("ingest/run/manifest.json", manifest).uri
    return runtime, manifest_uri, artifact_uris


def test_legacy_provenance_v1_still_loads_without_false_t08_capability(
    tmp_path: Path,
) -> None:
    """Keep historical recipes valid and make T09 opt-in failure explicit."""
    runtime, manifest_uri, _ = _stored_run(
        tmp_path, version="cognityx.ingest.provenance/v1"
    )
    resolved = resolve_source(runtime, manifest_uri)
    assert resolved.provenance[0].payload["schema_version"].endswith("/v1")
    with pytest.raises(V32HandoffUnavailableError, match="handoff unavailable"):
        resolved.require_v3_2()


def test_provenance_v2_falls_back_to_artifact_uris_without_loading_artifacts(
    tmp_path: Path,
) -> None:
    """Derive complete refs while leaving nonexistent parser/T08 objects unopened."""
    runtime, manifest_uri, expected = _stored_run(
        tmp_path, version="cognityx.ingest.provenance/v2"
    )
    document = resolve_source(runtime, manifest_uri).require_v3_2().documents[0]
    assert document.canonical_content_uri == expected["canonical_content"]
    assert document.source_graph_uri == expected["source_graph"]
    assert document.provenance_addresses_uri == expected["provenance_addresses"]


def test_run_manifest_direct_refs_load_and_are_recorded_in_selection(
    tmp_path: Path,
) -> None:
    """Prefer the producer ordering while retaining provenance as authority."""
    runtime, manifest_uri, _ = _stored_run(
        tmp_path,
        version="cognityx.ingest.provenance/v2",
        direct_refs=True,
    )
    resolved = resolve_source(runtime, manifest_uri)
    assert resolved.selection_manifest["dataforge_source_refs"] == [
        resolved.require_v3_2().documents[0].to_dict()
    ]


def test_direct_refs_and_provenance_disagreement_fails_closed(tmp_path: Path) -> None:
    """Reject competing URI truth rather than choosing or repairing one side."""
    runtime, manifest_uri, _ = _stored_run(
        tmp_path,
        version="cognityx.ingest.provenance/v2",
        direct_refs=True,
        conflict=True,
    )
    with pytest.raises(V32SourceConflictError, match="disagree"):
        resolve_source(runtime, manifest_uri)
