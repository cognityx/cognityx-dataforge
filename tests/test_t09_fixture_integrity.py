"""Verify the independently vendored T09 contract subset before consumption."""

from __future__ import annotations

import hashlib
import json

from t09_support import FIXTURE_ROOT


def test_vendored_t09_fixture_manifest_matches_exact_bytes() -> None:
    """Bind every vendored file to the reviewed upstream Ingest merge SHA."""
    manifest = json.loads(
        (FIXTURE_ROOT / "upstream_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["upstream_repository"] == "cognityx/cognityx-ingest"
    assert manifest["upstream_t08_merge_sha"] == (
        "dc0e8757392ed490118893c8956bb4f6b82f5e13"
    )
    assert set(manifest["files"]) == {
        "expected/canonical_content.json",
        "expected/source_graph.json",
        "expected/provenance_addresses.json",
        "segmentation_views/views.json",
        "dataforge/paragraph_qa_contract.json",
        "dataforge/composite_ku_contract.json",
    }
    for relative_path, expected in manifest["files"].items():
        assert hashlib.sha256((FIXTURE_ROOT / relative_path).read_bytes()).hexdigest() == expected
