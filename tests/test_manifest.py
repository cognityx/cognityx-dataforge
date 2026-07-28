from __future__ import annotations

from cognityx_dataforge.evidence import load_run_manifest


def test_manifest_schema():
    assert load_run_manifest({"schema": "cognityx.ingest.run-manifest/v1"})["schema"] == "cognityx.ingest.run-manifest/v1"

