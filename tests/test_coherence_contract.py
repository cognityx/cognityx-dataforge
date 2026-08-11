from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset
from cognityx_dataforge.cli import _runtime
from cognityx_dataforge.dataset import deduplicate_records, split_for_group
from cognityx_dataforge.execution import BuildIdentity
from cognityx_dataforge.source import resolve_source


class LineageClient:
    def __init__(self, jobs: JobRepository | None = None) -> None:
        self.jobs = jobs
        self.job_id: str | None = None

    def chat(self, **kwargs):
        if self.jobs is not None and self.job_id is not None:
            self.jobs.request_cancel(self.job_id)
        return {
            "id": "request-123",
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "instruction": "What is alpha?",
                        "answer": "Alpha is the first value.",
                    })
                }
            }],
        }

    def count_input_tokens(self, **kwargs):
        return 12


class RecordingStore:
    def __init__(self, store, writes: list[str]) -> None:
        self._store = store
        self._writes = writes

    def __getattr__(self, name):
        return getattr(self._store, name)

    def put_bytes(self, key, *args, **kwargs):
        self._writes.append(key)
        return self._store.put_bytes(key, *args, **kwargs)

    def put_json_idempotent(self, key, *args, **kwargs):
        self._writes.append(key)
        return self._store.put_json_idempotent(key, *args, **kwargs)


class RecordingRuntime:
    def __init__(self, runtime: StorageRuntime) -> None:
        self._runtime = runtime
        self.config = runtime.config
        self._backends = runtime._backends
        self.writes: list[str] = []
        self._dataset = RecordingStore(runtime.for_role("dataset"), self.writes)

    def for_role(self, role_name):
        if role_name == "dataset":
            return self._dataset
        return self._runtime.for_role(role_name)

    def for_profile(self, *args, **kwargs):
        return self._runtime.for_profile(*args, **kwargs)


def _fixture(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    artifact = runtime.for_role("artifact")
    evidence = {
        "evidence_id": "evidence-1",
        "document_id": "document-1",
        "source_asset_id": "asset-1",
        "page_number": 1,
        "text": "Alpha is the first value.",
        "char_start": 0,
        "char_end": 25,
        "anchor_id": "document-1:page-index:0",
        "block_id": "document-1:block:0",
        "source_sha256": "sha-1",
        "context_id": "context-1",
        "run_id": "ingest-run-1",
    }
    evidence_uri = artifact.put_bytes(
        "ingest/documents/document-1/evidence.jsonl",
        (json.dumps(evidence) + "\n").encode(),
        media_type="application/x-ndjson",
    ).uri
    provenance_uri = artifact.put_json(
        "ingest/documents/document-1/provenance.json",
        {
            "schema": "cognityx.ingest.provenance",
            "schema_version": "cognityx.ingest.provenance/v1",
            "document_id": "document-1",
            "source_asset": {
                "asset_id": "asset-1",
                "blob_sha256": "sha-1",
            },
            "evidence": [evidence],
            "pages": [{"page_id": "document-1:page-index:0"}],
            "blocks": [{"block_id": "document-1:block:0"}],
            "sections": [],
            "objects": [],
            "relations": [],
            "decisions": [],
            "unresolved": [],
            "parser": {"selected": "fixture"},
        },
    ).uri
    manifest_uri = artifact.put_json(
        "ingest/runs/ingest-run-1/manifest.json",
        {
            "schema": "cognityx.ingest.run",
            "run_id": "ingest-run-1",
            "context_id": "context-1",
            "source_assets": [{"asset_id": "asset-1", "sha256": "sha-1"}],
            "document_ids": ["document-1"],
            "evidence_refs": [evidence_uri],
            "provenance_refs": [provenance_uri],
        },
    ).uri
    config = tmp_path / "dataforge.toml"
    config.write_text(
        "[models.generator]\n"
        "model='fake'\n"
        "backend='fake'\n"
        "profile='test'\n"
        "max_output_tokens=16\n"
        "[splitting]\n"
        "seed='experiment-seed'\n",
        encoding="utf-8",
    )
    return runtime, manifest_uri, config


def test_default_storage_configuration_and_source_selection(tmp_path, monkeypatch):
    root = tmp_path / "configured-storage"
    config = tmp_path / "storage.toml"
    config.write_text(
        "[storage]\ndefault_profile='local-main'\n"
        "[storage.profiles.local-main]\ntype='filesystem'\n"
        f"root='{root}'\n"
        "[storage.roles.artifact]\nprofile='local-main'\nnamespace='artifacts'\n"
        "[storage.roles.dataset]\nprofile='local-main'\nnamespace='datasets'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("COGNITYX_STORAGE_CONFIG", str(config))
    args = type("Args", (), {"storage_root": None, "storage_config": None})()
    runtime = _runtime(args)
    assert runtime.config.source == str(config)

    fixture_runtime, manifest_uri, _ = _fixture(tmp_path)
    resolved = resolve_source(fixture_runtime, manifest_uri)
    assert resolved.selection_manifest["source_manifest_uri"] == manifest_uri
    assert resolved.evidence_anchor_ids["evidence-1"] == (
        "document-1:page-index:0",
        "document-1:block:0",
    )
    assert resolved.selection_manifest["provenance_checksums"]
    selection_uri = fixture_runtime.for_role("artifact").put_json(
        "selections/one.json",
        resolved.selection_manifest,
    ).uri
    assert resolve_source(fixture_runtime, selection_uri).source_manifest_uri == manifest_uri


def test_identity_lineage_selection_and_manifest_written_last(tmp_path):
    runtime, manifest_uri, config = _fixture(tmp_path)
    recording = RecordingRuntime(runtime)
    jobs = JobRepository(":memory:")
    result = build_dataset(
        manifest_uri,
        "comparison",
        "paragraph-qa",
        config,
        experiment_id="experiment-1",
        requested_run_id="run-experiment-step-1",
        runtime=recording,
        jobs=jobs,
        inference_client=LineageClient(),
    )
    assert result["run_id"] == "run-experiment-step-1"
    assert result["run_id"] != result["job_id"]
    assert result["variant_id"]
    assert jobs.get(result["job_id"]).state == "completed"

    dataset = runtime.for_role("dataset")
    manifest_store, manifest_key = _manifest(runtime, result["dataset_manifest_uri"])
    with manifest_store.open(manifest_key) as handle:
        manifest = json.load(handle)
    for key in ("experiment_id", "variant_id", "run_id", "job_id", "dataset_id"):
        assert manifest[key] == result[key]
    assert manifest["effective_configuration"]["split_seed"] == "experiment-seed"
    assert dataset.exists(
        f"dataforge/experiments/{result['experiment_id']}/variants/"
        f"{result['variant_id']}/runs/{result['run_id']}/input-selection.json"
    )
    with dataset.open(manifest_key.removesuffix("manifest.json") + "records.jsonl") as handle:
        record = json.loads(handle.readline())
    assert record["metadata"]["request_metadata"]["request_id"] == "request-123"
    assert record["metadata"]["source_anchor_ids"] == [
        "document-1:page-index:0",
        "document-1:block:0",
    ]
    assert record["metadata"]["enrichment_id"].startswith("enr-")
    assert manifest["provenance_refs"]
    publication_writes = [
        key for key in recording.writes
        if key.startswith(manifest_key.rsplit("/", 1)[0] + "/")
    ]
    assert publication_writes[-1] == manifest_key

    reused = build_dataset(
        manifest_uri,
        "comparison",
        "paragraph-qa",
        config,
        experiment_id="experiment-1",
        requested_run_id="run-experiment-step-1",
        runtime=recording,
        jobs=jobs,
        inference_client=LineageClient(),
    )
    assert reused["reused"] is True
    assert reused["dataset_manifest_uri"] == result["dataset_manifest_uri"]


def test_cancelled_build_keeps_run_artifacts_without_publication(tmp_path):
    runtime, manifest_uri, config = _fixture(tmp_path)
    jobs = JobRepository(":memory:")
    client = LineageClient(jobs)
    original_create = jobs.create

    def create(*args, **kwargs):
        record = original_create(*args, **kwargs)
        client.job_id = record.job_id
        return record

    jobs.create = create  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="cancelled"):
        build_dataset(
            manifest_uri,
            "cancelled",
            "paragraph-qa",
            config,
            experiment_id="experiment-cancel",
            runtime=runtime,
            jobs=jobs,
            inference_client=client,
        )
    record = jobs.get(client.job_id)
    assert record.state == "cancelled"
    assert not runtime.for_role("dataset").exists(
        "dataforge/experiments/experiment-cancel/variants"
    ) or not any(
        path.name == "manifest.json" and "datasets" in path.parts
        for path in (tmp_path / "storage").rglob("manifest.json")
    )


def test_group_splitting_and_exact_record_deduplication():
    records = [
        {
            "record_id": "one",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            "metadata": {"document_ids": ["doc-1"]},
        },
        {
            "record_id": "duplicate",
            "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
            "metadata": {"document_ids": ["doc-1"]},
        },
        {
            "record_id": "related",
            "messages": [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
            "metadata": {"document_ids": ["doc-1"]},
        },
    ]
    unique, duplicate_count = deduplicate_records(records, split_seed="seed")
    assert duplicate_count == 1
    assert len(unique) == 2
    assert len({record["split"] for record in unique}) == 1
    assert split_for_group("document_ids:doc-1", "seed") == unique[0]["split"]
    with pytest.raises(ValueError, match="experiment_id"):
        BuildIdentity.create(
            experiment_id="../escape",
            recipe="paragraph-qa",
            configuration_checksum="config",
            source_checksum="source",
        )


def test_cli_input_manifest_compatibility_alias(monkeypatch, capsys):
    from cognityx_dataforge import cli

    captured = {}

    def fake_build(source, dataset_name, recipe, config, **kwargs):
        captured.update(
            source=source,
            dataset_name=dataset_name,
            recipe=recipe,
            experiment_id=kwargs["experiment_id"],
        )
        return {"job_id": "job-1", "run_id": "run-1"}

    monkeypatch.setattr(cli, "build_dataset", fake_build)
    monkeypatch.setattr(cli, "_runtime", lambda args: object())
    monkeypatch.setattr(cli, "_jobs", lambda args: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cognityx-dataforge",
            "build",
            "paragraph-qa",
            "--input-manifest",
            "storage://local-main/artifacts/ingest/run.json",
            "--experiment-id",
            "experiment-1",
            "--config",
            "dataforge.toml",
        ],
    )
    with pytest.warns(FutureWarning, match="--input-manifest"):
        cli.main()
    assert captured == {
        "source": "storage://local-main/artifacts/ingest/run.json",
        "dataset_name": "experiment-1",
        "recipe": "paragraph-qa",
        "experiment_id": "experiment-1",
    }
    assert json.loads(capsys.readouterr().out)["job_id"] == "job-1"


def _manifest(runtime, uri):
    from cognityx_dataforge.source import resolve_storage_uri

    return resolve_storage_uri(runtime, uri, role_name="dataset")
