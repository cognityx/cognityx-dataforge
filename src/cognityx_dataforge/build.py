from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cognityx_jobs import JobRepository
from cognityx_storage import StorageClient, StorageConfig, StorageRuntime

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
from cognityx_dataforge.inference import StructuredAdapter
from cognityx_dataforge.knowledge import KnowledgeUnit, parse_knowledge_units
from cognityx_dataforge.models import DatasetRecord
from cognityx_dataforge.paragraphs import paragraph_spans
from cognityx_dataforge.recipes import normalize_recipe


def _runtime(root: str | Path | None, config_path: str | Path | None) -> StorageRuntime:
    if config_path:
        return StorageRuntime.load(config_file=config_path)
    return StorageRuntime.from_config(StorageConfig.built_in(root=root or "/tmp/cognityx-dataforge-storage"))


def resolve_storage_uri(runtime: StorageRuntime, uri: str, role_name: str = "artifact"):
    """Resolve both shared-scope and profile/namespace Storage URIs."""
    if not uri.startswith("storage://"):
        raise ValueError(f"Expected storage URI, got: {uri}")
    remainder = uri.removeprefix("storage://")
    first, separator, tail = remainder.partition("/")
    if first == "shared":
        profile = runtime.config.default_profile
        if profile is None:
            raise ValueError("Storage configuration has no default profile for shared URI")
        runtime.for_profile(profile, role_name=role_name)
        backend = runtime._backends[profile]  # Runtime owns the configured backend lifecycle.
        return StorageClient(backend).for_shared_data(), tail
    if first not in runtime.config.profiles:
        raise ValueError(f"Unknown storage URI profile: {first}")
    store = runtime.for_profile(first, role_name=role_name)
    key = tail if separator else ""
    namespace = store.namespace.strip("/")
    if namespace and key.startswith(namespace + "/"):
        key = key[len(namespace) + 1 :]
    return store, key


_store_for_uri = resolve_storage_uri


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in rows)


def build_dataset(
    input_manifest_uri: str,
    dataset_name: str,
    recipe: str | None,
    config_path: str | Path,
    *,
    runtime: StorageRuntime | None = None,
    jobs: JobRepository | None = None,
    inference_client: Any | None = None,
    storage_root: str | Path | None = None,
    storage_config: str | Path | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    runtime = runtime or _runtime(storage_root, storage_config)
    jobs = jobs or JobRepository(":memory:")
    config = DataForgeConfig.load(config_path)
    recipe = normalize_recipe(recipe, variant=variant)
    manifest_store, manifest_key = resolve_storage_uri(runtime, input_manifest_uri)
    with manifest_store.open(manifest_key) as handle:
        manifest = load_run_manifest(json.load(handle))
    dataset_id = deterministic_id(dataset_name, recipe)
    dataset_version = deterministic_id(dataset_name, recipe, checksum(manifest), checksum(asdict(config)), config.generator.model, config.validator.model if config.validator else "", *[f"{key}:{value}" for key, value in sorted(config.prompt_versions.items())])
    dataset_store = runtime.for_role("dataset")
    dataset_root = f"{dataset_id}/{dataset_version}"
    manifest_key = f"{dataset_root}/manifest.json"
    expected_source_checksum = checksum(manifest)
    expected_config_checksum = checksum(asdict(config))
    if dataset_store.exists(manifest_key):
        with dataset_store.open(manifest_key) as handle:
            existing = json.load(handle)
        records_store, records_key = resolve_storage_uri(runtime, existing["records_uri"], role_name="dataset")
        with records_store.open(records_key) as handle:
            records_bytes = handle.read()
        if (
            existing.get("source_manifest_checksum") != expected_source_checksum
            or existing.get("configuration_checksum") != expected_config_checksum
            or (existing.get("recipe") or {"v0": "paragraph-qa", "v1": "knowledge-unit-qa"}.get(existing.get("variant"))) != recipe
            or existing.get("generator") != asdict(config.generator)
            or checksum(records_bytes.decode("utf-8")) != existing.get("records_checksum")
        ):
            raise ValueError("Existing immutable dataset does not match this build identity")
        return {
            "run_id": existing["run_id"],
            "job_id": existing["job_id"],
            "dataset_id": existing["dataset_id"],
            "recipe": recipe,
            "record_count": existing["accepted_count"],
            "dataset_manifest_uri": dataset_store.uri(manifest_key),
            "reused": True,
        }
    job = jobs.create(
        deterministic_id(dataset_name, recipe, expected_source_checksum, expected_config_checksum),
        "dataforge.build",
        {"dataset_name": dataset_name, "recipe": recipe, "source_manifest_uri": input_manifest_uri},
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
        structured = StructuredAdapter(inference_client, GeneratorConfig(**asdict(config.generator)))
        validator_config = config.validator or config.generator
        validator = StructuredAdapter(inference_client, GeneratorConfig(**asdict(validator_config)))
        records: list[DatasetRecord] = []
        candidates: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        knowledge_units: list[KnowledgeUnit] = []
        prompt_dir = Path(__file__).with_name("prompts")
        prompt_template = prompt_dir.joinpath("v0_instruction_answer.txt").read_text(encoding="utf-8")
        for item in evidence:
            units = ()
            if recipe == "knowledge-unit-qa":
                discovery_prompt = prompt_dir.joinpath("v1_knowledge_unit_discovery.txt").read_text(encoding="utf-8")
                discovered = structured.ask(discovery_prompt + "\n\n" + item.text, "Return strict JSON for knowledge-unit discovery.")
                units = parse_knowledge_units(discovered, item.evidence_id, config.generator.model, config.prompt_versions["knowledge_unit"])
                knowledge_units.extend(units)
            else:
                units = (None,)
            spans = (
                [(start, end, text, None) for start, end, text in paragraph_spans(item.text)]
                if recipe == "paragraph-qa"
                else [(0, len(item.text), unit.canonical_statement, unit) for unit in units]
            )
            for start, end, text, unit in spans:
                if jobs.get(job.job_id).state == "cancellation_requested":
                    raise KeyboardInterrupt("DataForge build cancellation requested")
                candidate = {"evidence_id": item.evidence_id, "char_start": start, "char_end": end, "text": text}
                candidates.append(candidate)
                try:
                    if recipe == "knowledge-unit-qa":
                        generation_prompt = prompt_dir.joinpath("v1_knowledge_unit_generation.txt").read_text(encoding="utf-8")
                        generated = generator.generate(generation_prompt + "\n\n" + text)
                    else:
                        generated = generator.generate(prompt_template + "\n\n" + text)
                except Exception as exc:
                    rejections.append({**candidate, "reason": str(exc)})
                    continue
                if recipe == "knowledge-unit-qa":
                    validation_prompt = prompt_dir.joinpath("v1_knowledge_unit_validation.txt").read_text(encoding="utf-8")
                    validation_raw = validator.ask(validation_prompt + "\n\nEvidence:\n" + item.text + "\n\nInstruction:\n" + generated["instruction"] + "\n\nAnswer:\n" + generated["answer"], "Return accept or reject with reasons.")
                    validation_data = json.loads(validation_raw)
                    decision = validation_data.get("decision")
                    if decision not in {"accept", "reject"}:
                        raise ValueError("Validator decision must be accept or reject")
                    raw_reasons = validation_data.get("reasons", {})
                    validation = {
                        "knowledge_unit_id": unit.knowledge_unit_id,
                        "evidence_ids": list(unit.source_evidence_ids),
                        "decision": decision,
                        "reasons": {key: raw_reasons.get(key) for key in (
                            "answerability", "factual_support", "contradiction",
                            "missing_critical_information", "instruction_clarity",
                            "answer_completeness",
                        )},
                    }
                    validations.append(validation)
                    if decision != "accept":
                        rejections.append({**candidate, "knowledge_unit_id": unit.knowledge_unit_id, "reason": validation_data.get("reasons", "validator rejected")})
                        continue
                record = DatasetRecord(
                    record_id=deterministic_id(dataset_name, recipe, item.evidence_id, str(start), str(end), generated["instruction"], generated["answer"]),
                    messages=({"role": "user", "content": generated["instruction"]}, {"role": "assistant", "content": generated["answer"]}),
                    split=split_for_index(len(records)),
                    metadata={
                        "recipe": recipe,
                        "source_asset_ids": [item.source_asset_id] if item.source_asset_id else [],
                        "document_ids": [item.document_id],
                        "evidence_ids": [item.evidence_id],
                        "char_start": start,
                        "char_end": end,
                        "generator_model": config.generator.model,
                        "prompt_versions": dict(config.prompt_versions),
                        **({"knowledge_unit_id": unit.knowledge_unit_id} if unit is not None else {}),
                    },
                )
                records.append(record)
        records_payload = [record.to_dict() for record in records]
        candidates_bytes = _jsonl(candidates)
        records_bytes = _jsonl(records_payload)
        rejections_bytes = _jsonl(rejections)
        knowledge_units_bytes = _jsonl([unit.to_dict() for unit in knowledge_units])
        validations_bytes = _jsonl(validations)
        records_key = f"{dataset_root}/records.jsonl"
        manifest_payload = {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "recipe": recipe,
            "dataset_version": dataset_version,
            "source_manifest_uri": input_manifest_uri,
            "source_manifest_checksum": expected_source_checksum,
            "configuration_checksum": expected_config_checksum,
            "generator": asdict(config.generator),
            "schema_version": "cognityx.dataforge.dataset/v1",
            "prompt_versions": dict(config.prompt_versions),
            "knowledge_unit_count": len(knowledge_units),
            "candidate_count": len(candidates),
            "accepted_count": len(records),
            "rejected_count": len(rejections),
            "train_count": sum(record.split == "train" for record in records),
            "eval_count": sum(record.split == "eval" for record in records),
            "records_uri": dataset_store.uri(records_key),
            "records_checksum": checksum(records_bytes.decode("utf-8")),
            "knowledge_units_uri": dataset_store.uri(f"{dataset_root}/knowledge-units.jsonl"),
            "knowledge_units_checksum": checksum(knowledge_units_bytes.decode("utf-8")),
            "validations_uri": dataset_store.uri(f"{dataset_root}/validations.jsonl"),
            "validations_checksum": checksum(validations_bytes.decode("utf-8")),
            "validation_failure_counts": {"rejected": sum(item.get("decision") == "reject" for item in validations)},
            "run_id": job.job_id,
            "job_id": job.job_id,
            "created_at": time.time(),
        }
        dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
        dataset_store.put_bytes(records_key, records_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/candidates.jsonl", candidates_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/rejections.jsonl", rejections_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/knowledge-units.jsonl", knowledge_units_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/validations.jsonl", validations_bytes, media_type="application/x-ndjson")
        jobs.append_event(job.job_id, "build_completed", {"record_count": len(records), "dataset_id": dataset_id})
        jobs.set_state(job.job_id, "completed")
        dataset_store.put_bytes(f"{dataset_root}/run-events.jsonl", _jsonl(jobs.events(job.job_id)), media_type="application/x-ndjson")
        return {"run_id": job.job_id, "job_id": job.job_id, "dataset_id": dataset_id, "recipe": recipe, "record_count": len(records), "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json")}
    except KeyboardInterrupt:
        jobs.append_event(job.job_id, "build_cancelled", {})
        jobs.set_state(job.job_id, "cancelled")
        raise RuntimeError("DataForge build cancelled")
    except Exception as exc:
        jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
        jobs.set_state(job.job_id, "failed")
        raise
