from __future__ import annotations

from cognityx_dataforge.evidence import load_run_manifest


def test_manifest_schema():
    manifest = {
        "schema": "cognityx.ingest.run",
        "run_id": "run-1",
        "context_id": "ctx-1",
        "evidence_refs": [],
    }
    assert load_run_manifest(manifest)["schema"] == "cognityx.ingest.run"
