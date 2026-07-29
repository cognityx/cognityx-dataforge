from __future__ import annotations

import json
from pathlib import Path
import sys

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from cognityx_ingest import ExecutionContext, IngestService, PyPdfExtractor, SourceAssetRegistry
from cognityx_jobs import JobRepository
from cognityx_storage import LocalStorageBackend, StorageClient, StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset, resolve_storage_uri
from cognityx_dataforge.cli import main
from cognityx_dataforge.recipes import normalize_recipe


class FakeClient:
    calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps({"instruction": "Ask", "answer": "Answer"})}}]}

    def count_input_tokens(self, **kwargs):
        return len(kwargs["messages"][-1]["content"].split())


class KnowledgeFakeClient(FakeClient):
    validations = 0

    def chat(self, **kwargs):
        self.calls += 1
        prompt = kwargs["messages"][-1]["content"]
        if "Validate the instruction-answer" in prompt:
            self.validations += 1
            payload = {"decision": "reject" if self.validations == 2 else "accept", "reasons": {"factual_support": "checked"}}
        elif "Discover every" in prompt or "knowledge_units array" in prompt:
            payload = {"knowledge_units": [
                {"canonical_statement": "A rule applies", "supporting_facts": ["A rule applies"], "concepts": ["rule"]},
                {"canonical_statement": "A second fact applies", "supporting_facts": ["A second fact applies"], "concepts": ["fact"]},
            ]}
        else:
            payload = {"instruction": "Ask", "answer": "Answer"}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def config_file(tmp_path: Path) -> Path:
    config = tmp_path / "config.toml"
    config.write_text(
        '[models.generator]\nmodel="fake"\nbackend="fake"\nprofile="test"\nmax_output_tokens=32\n',
        encoding="utf-8",
    )
    return config


def knowledge_config_file(tmp_path: Path) -> Path:
    config = tmp_path / "knowledge-config.toml"
    config.write_text(
        'context_limit_tokens=256\n'
        '[models.generator]\nmodel="fake-generator"\nbackend="fake"\nprofile="test"\nmax_output_tokens=32\n'
        '[models.validator]\nmodel="fake-validator"\nbackend="fake"\nprofile="test"\nmax_output_tokens=32\n'
        '[prompt_versions]\nknowledge_unit="1.0"\ngeneration="1.0"\nvalidation="1.0"\n'
        ,
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
    fake = FakeClient()
    result = build_dataset(
        manifest_uri,
        "demo",
        "v0",
        config_file(tmp_path),
        runtime=runtime,
        jobs=jobs,
        inference_client=fake,
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

    reused = build_dataset(
        manifest_uri,
        "demo",
        "v0",
        config_file(tmp_path),
        runtime=runtime,
        jobs=jobs,
        inference_client=fake,
    )
    assert reused["reused"] is True
    assert reused["dataset_manifest_uri"] == result["dataset_manifest_uri"]
    assert fake.calls == 1


def test_resolve_storage_uri_supports_shared_and_profile_namespace(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    shared_store, shared_key = resolve_storage_uri(runtime, "storage://shared/ingest/run/manifest.json")
    profile_store, profile_key = resolve_storage_uri(runtime, "storage://local-main/artifacts/ingest/run/evidence.jsonl")
    assert shared_key == "ingest/run/manifest.json"
    assert profile_key == "ingest/run/evidence.jsonl"
    assert shared_store.backend_name == profile_store.backend_name


def test_recipe_aliases_and_legacy_manifest_readability():
    assert normalize_recipe("v0") == "paragraph-qa"
    assert normalize_recipe("v1") == "knowledge-unit-qa"
    assert normalize_recipe(variant="v0") == "paragraph-qa"
    assert normalize_recipe("knowledge-unit-probed-qa") == "knowledge-unit-probed-qa"


def test_knowledge_unit_recipe_discovers_validates_and_rejects(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    artifacts = runtime.for_role("artifact")
    evidence = {"evidence_id": "e1", "document_id": "doc-1", "page_number": 1, "text": "A rule applies. A second fact applies.", "char_start": 0, "char_end": 38, "source_asset_id": "asset-1", "bundle_id": "bundle-1", "context_id": "ctx-1", "sequence_number": 0, "source_sha256": "sha-1", "run_id": "run-ku"}
    evidence_uri = artifacts.put_bytes("ingest/documents/doc-1/evidence.jsonl", (json.dumps(evidence) + "\n").encode(), media_type="application/x-ndjson").uri
    manifest_uri = artifacts.put_json("ingest/runs/run-ku/manifest.json", {"schema": "cognityx.ingest.run", "run_id": "run-ku", "context_id": "ctx-1", "source_assets": [{"asset_id": "asset-1", "sha256": "sha-1"}], "document_ids": ["doc-1"], "evidence_refs": [evidence_uri]}).uri
    fake = KnowledgeFakeClient()
    result = build_dataset(manifest_uri, "ku", "knowledge-unit-qa", knowledge_config_file(tmp_path), runtime=runtime, inference_client=fake)
    assert result["recipe"] == "knowledge-unit-qa"
    dataset = runtime.for_role("dataset")
    version = result["dataset_manifest_uri"].rsplit("/", 2)[-2]
    with dataset.open(f"{result['dataset_id']}/{version}/manifest.json") as handle:
        payload = json.load(handle)
    assert payload["recipe"] == "knowledge-unit-qa"
    assert payload["knowledge_unit_count"] == 2
    assert payload["accepted_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["dataset_name"] == "ku"
    for stage in ("discovery", "generation", "validation", "finalization"):
        assert dataset.exists(f"{result['dataset_id']}/{version}/checkpoints/{stage}.json")
    reused = build_dataset(manifest_uri, "ku", "knowledge-unit-qa", knowledge_config_file(tmp_path), runtime=runtime, inference_client=fake)
    assert reused["reused"] is True


def test_knowledge_unit_budget_rejection_is_structured(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    artifacts = runtime.for_role("artifact")
    evidence = {"evidence_id": "e1", "document_id": "doc-1", "page_number": 1, "text": "oversized evidence", "char_start": 0, "char_end": 18, "context_id": "ctx-1", "run_id": "run-budget"}
    evidence_uri = artifacts.put_bytes("ingest/documents/doc-1/evidence.jsonl", (json.dumps(evidence) + "\n").encode(), media_type="application/x-ndjson").uri
    manifest_uri = artifacts.put_json("ingest/runs/run-budget/manifest.json", {"schema": "cognityx.ingest.run", "run_id": "run-budget", "context_id": "ctx-1", "document_ids": ["doc-1"], "evidence_refs": [evidence_uri]}).uri
    config = knowledge_config_file(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("context_limit_tokens=256", "context_limit_tokens=1"), encoding="utf-8")
    result = build_dataset(manifest_uri, "budget", "knowledge-unit-qa", config, runtime=runtime, inference_client=KnowledgeFakeClient())
    dataset = runtime.for_role("dataset")
    version = result["dataset_manifest_uri"].rsplit("/", 2)[-2]
    with dataset.open(f"{result['dataset_id']}/{version}/rejections.jsonl") as handle:
        rejection = json.loads(handle.readline())
    assert rejection["reason"] == "token_budget_exceeded"
    assert "input_tokens" in rejection


def test_real_ingest_to_dataforge_export(tmp_path: Path):
    root = tmp_path / "storage"
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=root))
    registry = SourceAssetRegistry.load(runtime=runtime)
    ingest_storage = StorageClient(LocalStorageBackend(root)).for_shared_data()
    service = IngestService(ingest_storage, extractor=PyPdfExtractor(), registry=registry)
    pdf = tmp_path / "fixture.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = writer._add_object(
        DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
    )
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (DataForge integration) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    with pdf.open("wb") as handle:
        writer.write(handle)
    context = ExecutionContext(
        run_id="real-ingest-run",
        correlation_id="real-ingest-correlation",
        principal_id="test-owner",
        tenant_id="test-tenant",
    )
    ingest_result = service.ingest_path(pdf, context=context, registry=registry)
    assert ingest_result.run_id == "real-ingest-run"
    fake = KnowledgeFakeClient()
    result = build_dataset(
        ingest_result.run_manifest_uri,
        "real-ingest-ku",
        "knowledge-unit-qa",
        knowledge_config_file(tmp_path),
        runtime=runtime,
        inference_client=fake,
    )
    assert result["record_count"] >= 1
    output = tmp_path / "real-export.jsonl"
    old_argv = sys.argv
    try:
        sys.argv = ["cognityx-dataforge", "dataset", "export", result["dataset_manifest_uri"], "--storage-root", str(root), "--output", str(output)]
        main()
    finally:
        sys.argv = old_argv
    assert output.read_text(encoding="utf-8").strip()
