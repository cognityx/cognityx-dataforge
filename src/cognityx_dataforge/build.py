from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cognityx_jobs import JobRepository
from cognityx_storage import StorageClient
from cognityx_storage.local import LocalStorageBackend
from cognityx_dataforge.config import DataForgeConfig
from cognityx_dataforge.dataset import checksum, deterministic_id, split_for_index
from cognityx_dataforge.evidence import evidence_availability, load_evidence_records, load_run_manifest, validate_context
from cognityx_dataforge.inference import GeneratorAdapter, GeneratorConfig
from cognityx_dataforge.models import DatasetRecord
from cognityx_dataforge.paragraphs import paragraph_spans


def _storage_client(root: str | Path | None = None) -> StorageClient:
    return StorageClient(LocalStorageBackend(root or "/tmp/cognityx-dataforge-storage"))


def build_dataset(
    input_manifest_uri: str,
    dataset_name: str,
    variant: str,
    config_path: str | Path,
    *,
    storage: StorageClient | None = None,
    jobs: JobRepository | None = None,
    inference_client: Any | None = None,
) -> dict[str, Any]:
    storage = storage or _storage_client()
    jobs = jobs or JobRepository(":memory:")
    config = DataForgeConfig.load(config_path)
    manifest_key = input_manifest_uri.removeprefix("storage://")
    with storage.open(manifest_key) as handle:
        manifest = load_run_manifest(json.load(handle))
    evidence_payload = manifest.get("evidence_payload")
    if not evidence_payload:
        raise ValueError("Run manifest must include evidence_payload for V0")
    evidence = load_evidence_records(evidence_payload)
    validate_context(manifest, evidence)
    evidence_availability(evidence)
    job = jobs.create(
        deterministic_id(dataset_name, variant, checksum(manifest), checksum(asdict(config))),
        "dataforge.build",
        {"dataset_name": dataset_name, "variant": variant},
    )
    if inference_client is None:
        from cognityx_inference.client import CognityxInferenceClient

        inference_client = CognityxInferenceClient()
    generator = GeneratorAdapter(inference_client, GeneratorConfig(**asdict(config.generator)))
    records: list[DatasetRecord] = []
    candidates: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for evidence_index, item in enumerate(evidence):
        jobs.append_event(job.job_id, "evidence_loaded", {"index": evidence_index, "evidence_id": item.evidence_id})
        for start, end, text in paragraph_spans(item.text):
            if jobs.get(job.job_id).state == "cancellation_requested":
                jobs.append_event(job.job_id, "cancel_observed", {"evidence_id": item.evidence_id})
                break
            candidate = {"evidence_id": item.evidence_id, "char_start": start, "char_end": end, "text": text}
            candidates.append(candidate)
            prompt = Path(__file__).with_name("prompts").joinpath("v0_instruction_answer.txt").read_text(encoding="utf-8") + "\n\n" + text
            try:
                generated = generator.generate(prompt)
            except Exception as exc:
                rejections.append({**candidate, "reason": str(exc)})
                continue
            record_id = deterministic_id(dataset_name, variant, item.evidence_id, str(start), str(end), generated["instruction"], generated["answer"])
            record = DatasetRecord(
                record_id=record_id,
                messages=(
                    {"role": "user", "content": generated["instruction"]},
                    {"role": "assistant", "content": generated["answer"]},
                ),
                split=split_for_index(len(records)),
                metadata={
                    "variant": variant,
                    "source_asset_ids": [item.source_asset_id] if item.source_asset_id else [],
                    "document_ids": [item.document_id],
                    "evidence_ids": [item.evidence_id],
                    "char_start": start,
                    "char_end": end,
                    "generator_model": config.generator.model,
                    "prompt_version": config.prompt_version,
                },
            )
            records.append(record)
        jobs.append_event(job.job_id, "evidence_completed", {"evidence_id": item.evidence_id})
    dataset_version = deterministic_id(dataset_name, variant, checksum(manifest), checksum(asdict(config)), config.generator.model, config.prompt_version)
    dataset_id = deterministic_id(dataset_name, variant)
    records_payload = [r.to_dict() for r in records]
    candidates_payload = candidates
    rejections_payload = rejections
    run_events_payload = jobs.events(job.job_id)
    manifest_payload = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "variant": variant,
        "dataset_version": dataset_version,
        "source_manifest_uri": input_manifest_uri,
        "source_manifest_checksum": checksum(manifest),
        "configuration_checksum": checksum(asdict(config)),
        "generator": asdict(config.generator),
        "prompt_version": config.prompt_version,
        "accepted_count": len(records),
        "rejected_count": len(rejections),
        "train_count": sum(1 for record in records if record.split == "train"),
        "eval_count": sum(1 for record in records if record.split == "eval"),
        "records_uri": storage.uri(f"datasets/{dataset_id}/{dataset_version}/records.jsonl"),
        "records_checksum": checksum(records_payload),
        "run_id": job.job_id,
        "job_id": job.job_id,
        "created_at": time.time(),
    }
    dataset_root = f"datasets/{dataset_id}/{dataset_version}"
    storage.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
    storage.put_bytes(f"{dataset_root}/records.jsonl", ("\n".join(json.dumps(row, sort_keys=True) for row in records_payload) + ("\n" if records_payload else "")).encode("utf-8"), media_type="application/x-ndjson")
    storage.put_bytes(f"{dataset_root}/candidates.jsonl", ("\n".join(json.dumps(row, sort_keys=True) for row in candidates_payload) + ("\n" if candidates_payload else "")).encode("utf-8"), media_type="application/x-ndjson")
    storage.put_bytes(f"{dataset_root}/rejections.jsonl", ("\n".join(json.dumps(row, sort_keys=True) for row in rejections_payload) + ("\n" if rejections_payload else "")).encode("utf-8"), media_type="application/x-ndjson")
    storage.put_bytes(f"{dataset_root}/run-events.jsonl", ("\n".join(json.dumps(row, sort_keys=True) for row in run_events_payload) + ("\n" if run_events_payload else "")).encode("utf-8"), media_type="application/x-ndjson")
    return {
        "run_id": job.job_id,
        "job_id": job.job_id,
        "dataset_id": dataset_id,
        "variant": variant,
        "record_count": len(records),
        "dataset_manifest_uri": storage.uri(f"{dataset_root}/manifest.json"),
    }
