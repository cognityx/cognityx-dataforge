from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.config import DataForgeConfig
from cognityx_dataforge.dataset import checksum, deterministic_id, split_for_index
from cognityx_dataforge.evidence import (
    combine_evidence,
    evidence_availability,
    load_evidence_jsonl,
    load_run_manifest,
    validate_context,
)
from cognityx_dataforge.inference import GeneratorAdapter, GeneratorConfig, load_inference_client
from cognityx_dataforge.models import DatasetRecord
from cognityx_dataforge.paragraphs import paragraph_spans


def _runtime(root: str | Path | None, config_path: str | Path | None) -> StorageRuntime:
    if config_path:
        return StorageRuntime.load(config_file=config_path)
    return StorageRuntime.from_config(StorageConfig.built_in(root=root or "/tmp/cognityx-dataforge-storage"))


def _store_for_uri(runtime: StorageRuntime, uri: str, role_name: str = "artifact"):
    if not uri.startswith("storage://"):
        raise ValueError(f"Expected storage URI, got: {uri}")
    remainder = uri.removeprefix("storage://")
    profile, _, key = remainder.partition("/")
    store = runtime.for_profile(profile, role_name=role_name)
    namespace = store.namespace.strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    return store, key


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in rows)


def build_dataset(
    input_manifest_uri: str,
    dataset_name: str,
    variant: str,
    config_path: str | Path,
    *,
    runtime: StorageRuntime | None = None,
    jobs: JobRepository | None = None,
    inference_client: Any | None = None,
    storage_root: str | Path | None = None,
    storage_config: str | Path | None = None,
) -> dict[str, Any]:
    runtime = runtime or _runtime(storage_root, storage_config)
    jobs = jobs or JobRepository(":memory:")
    config = DataForgeConfig.load(config_path)
    manifest_store, manifest_key = _store_for_uri(runtime, input_manifest_uri)
    with manifest_store.open(manifest_key) as handle:
        manifest = load_run_manifest(json.load(handle))
    job = jobs.create(
        deterministic_id(dataset_name, variant, checksum(manifest), checksum(asdict(config))),
        "dataforge.build",
        {"dataset_name": dataset_name, "variant": variant, "source_manifest_uri": input_manifest_uri},
    )
    jobs.set_state(job.job_id, "running")
    jobs.append_event(job.job_id, "build_started", {"run_id": manifest["run_id"]})
    try:
        evidence_groups = []
        for evidence_ref in manifest["evidence_refs"]:
            evidence_store, evidence_key = _store_for_uri(runtime, evidence_ref)
            with evidence_store.open(evidence_key) as handle:
                group = load_evidence_jsonl(handle)
            evidence_groups.append(group)
            jobs.append_event(job.job_id, "evidence_loaded", {"evidence_ref": evidence_ref, "count": len(group)})
        evidence = combine_evidence(evidence_groups)
        validate_context(manifest, evidence)
        evidence_availability(evidence)
        if inference_client is None:
            inference_client = load_inference_client()
        generator = GeneratorAdapter(inference_client, GeneratorConfig(**asdict(config.generator)))
        records: list[DatasetRecord] = []
        candidates: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        prompt_template = Path(__file__).with_name("prompts").joinpath("v0_instruction_answer.txt").read_text(encoding="utf-8")
        for item in evidence:
            for start, end, text in paragraph_spans(item.text):
                if jobs.get(job.job_id).state == "cancellation_requested":
                    raise KeyboardInterrupt("DataForge build cancellation requested")
                candidate = {"evidence_id": item.evidence_id, "char_start": start, "char_end": end, "text": text}
                candidates.append(candidate)
                try:
                    generated = generator.generate(prompt_template + "\n\n" + text)
                except Exception as exc:
                    rejections.append({**candidate, "reason": str(exc)})
                    continue
                record = DatasetRecord(
                    record_id=deterministic_id(dataset_name, variant, item.evidence_id, str(start), str(end), generated["instruction"], generated["answer"]),
                    messages=({"role": "user", "content": generated["instruction"]}, {"role": "assistant", "content": generated["answer"]}),
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
        dataset_id = deterministic_id(dataset_name, variant)
        dataset_version = deterministic_id(dataset_name, variant, checksum(manifest), checksum(asdict(config)), config.generator.model, config.prompt_version)
        records_payload = [record.to_dict() for record in records]
        candidates_bytes = _jsonl(candidates)
        records_bytes = _jsonl(records_payload)
        rejections_bytes = _jsonl(rejections)
        dataset_store = runtime.for_role("dataset")
        dataset_root = f"{dataset_id}/{dataset_version}"
        records_key = f"{dataset_root}/records.jsonl"
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
            "train_count": sum(record.split == "train" for record in records),
            "eval_count": sum(record.split == "eval" for record in records),
            "records_uri": dataset_store.uri(records_key),
            "records_checksum": checksum(records_bytes.decode("utf-8")),
            "run_id": job.job_id,
            "job_id": job.job_id,
            "created_at": time.time(),
        }
        dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
        dataset_store.put_bytes(records_key, records_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/candidates.jsonl", candidates_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/rejections.jsonl", rejections_bytes, media_type="application/x-ndjson")
        jobs.append_event(job.job_id, "build_completed", {"record_count": len(records), "dataset_id": dataset_id})
        jobs.set_state(job.job_id, "completed")
        dataset_store.put_bytes(f"{dataset_root}/run-events.jsonl", _jsonl(jobs.events(job.job_id)), media_type="application/x-ndjson")
        return {"run_id": job.job_id, "job_id": job.job_id, "dataset_id": dataset_id, "variant": variant, "record_count": len(records), "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json")}
    except KeyboardInterrupt:
        jobs.append_event(job.job_id, "build_cancelled", {})
        jobs.set_state(job.job_id, "cancelled")
        raise RuntimeError("DataForge build cancelled")
    except Exception as exc:
        jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
        jobs.set_state(job.job_id, "failed")
        raise
