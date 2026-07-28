from __future__ import annotations

import json
from pathlib import Path
import sys

from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset
from cognityx_dataforge.cli import main


class FakeClient:
    def chat(self, **kwargs):
        return {"choices": [{"message": {"content": json.dumps({"instruction": "Ask", "answer": "Answer"})}}]}


def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.generator]\nmodel="fake"\nbackend="fake"\nprofile="test"\nmax_output_tokens=32\n',
        encoding="utf-8",
    )
    return config


def test_build_dataset_with_real_ingest_manifest(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    artifacts = runtime.for_role("artifact")
    evidence = {
        "evidence_id": "e1",
        "document_id": "doc-1",
        "page_number": 1,
        "text": "alpha beta",
        "char_start": 0,
        "char_end": 10,
        "source_asset_id": "asset-1",
        "bundle_id": "bundle-1",
        "context_id": "ctx-1",
        "sequence_number": 0,
        "source_sha256": "sha-1",
        "run_id": "run-1",
        "schema_version": "cognityx.ingest.evidence/v2",
    }
    evidence_uri = artifacts.put_bytes(
        "ingest/documents/doc-1/evidence.jsonl",
        (json.dumps(evidence) + "\n").encode(),
        media_type="application/x-ndjson",
    ).uri
    manifest = {
        "schema": "cognityx.ingest.run",
        "schema_version": "cognityx.ingest.run/v1",
        "run_id": "run-1",
        "context_id": "ctx-1",
        "source_assets": [{"asset_id": "asset-1", "bundle_id": "bundle-1", "sha256": "sha-1"}],
        "document_ids": ["doc-1"],
        "evidence_refs": [evidence_uri],
    }
    manifest_uri = artifacts.put_json("ingest/runs/run-1/manifest.json", manifest).uri
    jobs = JobRepository(":memory:")
    result = build_dataset(
        manifest_uri,
        "demo",
        "v0",
        config_file(tmp_path),
        runtime=runtime,
        jobs=jobs,
        inference_client=FakeClient(),
    )
    assert result["record_count"] == 1
    job = jobs.get(result["job_id"])
    assert job.state == "completed"
    assert [event["event"] for event in jobs.events(job.job_id)] == [
        "build_started",
        "evidence_loaded",
        "build_completed",
    ]
    dataset = runtime.for_role("dataset")
    with dataset.open(f"{result['dataset_id']}/{result['dataset_manifest_uri'].rsplit('/', 2)[-2]}/manifest.json") as handle:
        dataset_manifest = json.load(handle)
    assert dataset_manifest["records_uri"].startswith("storage://")
    output = tmp_path / "export.jsonl"
    old_argv = sys.argv
    try:
        sys.argv = [
            "cognityx-dataforge",
            "dataset",
            "export",
            result["dataset_manifest_uri"],
            "--storage-root",
            str(tmp_path / "storage"),
            "--output",
            str(output),
        ]
        main()
    finally:
        sys.argv = old_argv
    assert json.loads(output.read_text(encoding="utf-8"))["messages"][0]["content"] == "Ask"
