from __future__ import annotations

from cognityx_ingest import EnrichmentIdentity

from cognityx_dataforge.evidence import load_run_manifest


def test_manifest_schema():
    manifest = {
        "schema": "cognityx.ingest.run",
        "run_id": "run-1",
        "context_id": "ctx-1",
        "evidence_refs": [],
    }
    assert load_run_manifest(manifest)["schema"] == "cognityx.ingest.run"


def test_enrichment_identity_is_stable_and_configuration_sensitive():
    values = {
        "source_content_hash": "sha-1",
        "source_anchor_ids": ("page-2", "page-1"),
        "representation_type": "training-record",
        "generation_method": "knowledge-unit-qa",
        "model_version": "approved/model",
    }
    first = EnrichmentIdentity.create(**values, configuration={"temperature": 0})
    reordered = EnrichmentIdentity.create(
        **{**values, "source_anchor_ids": ("page-1", "page-2")},
        configuration={"temperature": 0},
    )
    changed = EnrichmentIdentity.create(
        **values,
        configuration={"temperature": 0.2},
    )

    assert first.enrichment_id == reordered.enrichment_id
    assert first.enrichment_id != changed.enrichment_id
