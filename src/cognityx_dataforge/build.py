from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cognityx_ingest import EnrichmentIdentity
from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.config import DataForgeConfig
from cognityx_dataforge.dataset import (
    checksum,
    deduplicate_records,
    deterministic_id,
)
from cognityx_dataforge.evidence import (
    combine_evidence,
    evidence_availability,
    load_evidence_jsonl,
    validate_context,
)
from cognityx_dataforge.execution import BuildIdentity, load_job_repository
from cognityx_dataforge.inference import (
    GeneratorAdapter,
    GeneratorConfig,
    InferenceClientPool,
    StructuredAdapter,
    TokenBudgetError,
    normalized_error_category,
)
from cognityx_dataforge.knowledge import KnowledgeUnit, parse_knowledge_units
from cognityx_dataforge.models import DatasetRecord
from cognityx_dataforge.paragraphs import paragraph_spans
from cognityx_dataforge.qualification import QualificationPipeline
from cognityx_dataforge.recipes import normalize_recipe
from cognityx_dataforge.source import resolve_source, resolve_storage_uri


def _source_anchor_ids(evidence: list[Any]) -> list[str]:
    return list(dict.fromkeys(
        str(value)
        for item in evidence
        for value in (item.anchor_id, item.block_id)
        if value
    ))


def _enrichment_id(
    evidence: list[Any],
    *,
    representation_type: str,
    generation_method: str,
    model_version: str | None,
    configuration: dict[str, Any],
) -> str:
    source_hashes = sorted({item.source_sha256 for item in evidence if item.source_sha256})
    return EnrichmentIdentity.create(
        source_content_hash=checksum(source_hashes),
        source_anchor_ids=tuple(_source_anchor_ids(evidence)),
        representation_type=representation_type,
        generation_method=generation_method,
        model_version=model_version,
        configuration=configuration,
    ).enrichment_id


def _runtime(root: str | Path | None, config_path: str | Path | None) -> StorageRuntime:
    if root is not None:
        warnings.warn(
            "storage_root is deprecated; configure Cognityx Storage instead.",
            FutureWarning,
            stacklevel=2,
        )
        return StorageRuntime.from_config(StorageConfig.built_in(root=root))
    if config_path:
        return StorageRuntime.load(config_file=config_path)
    return StorageRuntime.load()


_store_for_uri = resolve_storage_uri


def _identity_fields(identity: BuildIdentity) -> dict[str, str]:
    return identity.fields()


def _persist_run_events(
    store: Any,
    run_root: str,
    jobs: JobRepository,
    job_id: str,
) -> None:
    store.put_bytes(
        f"{run_root}/run-events.jsonl",
        _jsonl(jobs.events(job_id)),
        media_type="application/x-ndjson",
    )


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n" for row in rows)


def _read_jsonl(store: Any, key: str) -> list[dict[str, Any]]:
    if not store.exists(key):
        return []
    with store.open(key) as handle:
        return [json.loads(line) for line in handle.read().decode().splitlines() if line.strip()]


def _checkpoint_key(dataset_root: str, stage: str) -> str:
    return f"{dataset_root}/checkpoints/{stage}.json"


def _write_checkpoint(store: Any, dataset_root: str, stage: str, identity: dict[str, Any], artifacts: dict[str, str], row_count: int) -> None:
    store.put_json_idempotent(_checkpoint_key(dataset_root, stage), {
        "stage": stage, "status": "completed", "identity": identity,
        "artifacts": artifacts, "row_count": row_count,
    })


def _completed_checkpoint(store: Any, dataset_root: str, stage: str, identity: dict[str, Any], artifacts: dict[str, str]) -> bool:
    key = _checkpoint_key(dataset_root, stage)
    if not store.exists(key):
        return False
    with store.open(key) as handle:
        checkpoint = json.load(handle)
    if checkpoint.get("status") != "completed" or checkpoint.get("identity") != identity:
        return False
    expected_artifacts = artifacts or checkpoint.get("artifacts", {})
    for artifact_key, expected_checksum in expected_artifacts.items():
        if not store.exists(artifact_key):
            return False
        with store.open(artifact_key) as handle:
            actual = checksum(handle.read().decode("utf-8"))
        if actual != expected_checksum:
            return False
    return True


def _check_cancel(jobs: JobRepository, job_id: str, stage: str) -> None:
    if jobs.get(job_id).state == "cancellation_requested":
        jobs.append_event(job_id, "stage_cancelled", {"stage": stage})
        jobs.set_state(job_id, "cancelled")
        raise RuntimeError("DataForge build cancelled")


def _role_config(config: DataForgeConfig, name: str) -> GeneratorConfig:
    value = getattr(config, name) or config.generator
    return GeneratorConfig(**asdict(value))


def _qualified_models(config: DataForgeConfig) -> dict[str, dict[str, Any]]:
    return {
        "generator": asdict(config.generator),
        **{
            name: asdict(getattr(config, name) or config.validator or config.generator)
            for name in ("answer_requirements", "source_answerability", "reference_qualification")
        },
    }


def _probed_identity(
    dataset_name: str,
    identity: BuildIdentity,
    manifest: dict[str, Any],
    config: DataForgeConfig,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        **_identity_fields(identity),
        "source_manifest_checksum": checksum(manifest), "configuration_checksum": checksum(asdict(config)),
        "recipe": "knowledge-unit-probed-qa", "probing": {"probes_per_unit": config.probes_per_unit, "include_classes": list(config.include_classes), "known_sample_rate": config.known_sample_rate},
        "prompt_versions": dict(config.prompt_versions),
    }


def _probed_json(raw: str, *, required: tuple[str, ...]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or any(not value.get(key) for key in required):
        raise ValueError("Malformed structured model output")
    return value


def _build_knowledge_unit_probed(*, manifest: dict[str, Any], dataset_name: str, config: DataForgeConfig, runtime: StorageRuntime, jobs: JobRepository, inference_client: Any, dataset_store: Any, dataset_root: str, identity: BuildIdentity, input_manifest_uri: str) -> dict[str, Any]:
    job_id = identity.job_id
    dataset_id = identity.dataset_id
    dataset_version = identity.dataset_version
    prompt_dir = Path(__file__).with_name("prompts")
    keys = {name: f"{dataset_root}/{name}.jsonl" for name in ("knowledge-units", "probes", "student-responses", "probe-judgments", "selected-units", "candidates", "validations", "records", "rejections")}
    evidence_groups = []
    for ref in manifest["evidence_refs"]:
        store, key = resolve_storage_uri(runtime, ref)
        with store.open(key) as handle:
            evidence_groups.append(load_evidence_jsonl(handle))
    evidence = combine_evidence(evidence_groups)
    validate_context(manifest, evidence)
    evidence_availability(evidence)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    checkpoint_identity = _probed_identity(dataset_name, identity, manifest, config)
    pool = inference_client if isinstance(inference_client, InferenceClientPool) else InferenceClientPool(config=config, injected_client=inference_client)
    pool.set_lineage(dataset_id=dataset_id, dataset_version=dataset_version, run_id=identity.run_id, job_id=job_id, recipe="knowledge-unit-probed-qa", data_classification=config.data_classification)
    calls: list[dict[str, Any]] = []
    rejections = _read_jsonl(dataset_store, keys["rejections"])

    def checkpoint(stage: str, artifacts: list[str], rows: int) -> None:
        checksums = {}
        for key in artifacts:
            with dataset_store.open(key) as handle:
                checksums[key] = checksum(handle.read().decode())
        _write_checkpoint(dataset_store, dataset_root, stage, checkpoint_identity, checksums, rows)

    units = [KnowledgeUnit.from_dict(row) for row in _read_jsonl(dataset_store, keys["knowledge-units"])]
    probes = _read_jsonl(dataset_store, keys["probes"])
    _check_cancel(jobs, job_id, "discovery")
    jobs.append_event(job_id, "stage_started", {"stage": "discovery"})
    if not _completed_checkpoint(dataset_store, dataset_root, "discovery", checkpoint_identity, {}):
        discovery = StructuredAdapter(pool, _role_config(config, "knowledge_unit"))
        for item in evidence:
            _check_cancel(jobs, job_id, "discovery")
            try:
                raw = discovery.ask_budgeted("Discover knowledge units from this evidence:\n" + item.text, "Return JSON with knowledge_units.", context_limit=config.context_limit_tokens, role="knowledge_unit", prompt_version=config.prompt_versions.get("knowledge_unit", "1.0"), evidence_ids=[item.evidence_id], calls=calls)
                units.extend(parse_knowledge_units(raw, item.evidence_id, discovery.config.model, config.prompt_versions.get("knowledge_unit", "1.0")))
            except (TokenBudgetError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                rejections.append({"stage": "discovery", "evidence_ids": [item.evidence_id], "reason": normalized_error_category(exc), "error": str(exc)})
        for unit in units:
            cited = [evidence_by_id[eid] for eid in unit.source_evidence_ids if eid in evidence_by_id]
            for index in range(max(0, config.probes_per_unit)):
                probe_id = deterministic_id(unit.knowledge_unit_id, str(index), "probe")
                probes.append({"probe_id": probe_id, "knowledge_unit_id": unit.knowledge_unit_id, "question_index": index, "question": "", "source_asset_ids": sorted({x.source_asset_id for x in cited if x.source_asset_id}), "document_ids": sorted({x.document_id for x in cited}), "evidence_ids": list(unit.source_evidence_ids)})
        probe_generator = StructuredAdapter(pool, _role_config(config, "probe_generator"))
        for probe in probes:
            if probe["question"]:
                continue
            unit = next(unit for unit in units if unit.knowledge_unit_id == probe["knowledge_unit_id"])
            try:
                raw = probe_generator.ask_budgeted(prompt_dir.joinpath("v2_probe_generation.txt").read_text(encoding="utf-8") + "\n" + unit.canonical_statement, "Return JSON with question.", context_limit=config.context_limit_tokens, role="probe_generator", prompt_version=config.prompt_versions.get("probe_generation", "2.0"), evidence_ids=list(unit.source_evidence_ids), calls=calls)
                probe["question"] = _probed_json(raw, required=("question",))["question"]
                probe["model"] = asdict(probe_generator.config)
                probe["prompt_version"] = config.prompt_versions.get("probe_generation", "2.0")
                probe["request_metadata"] = dict(calls[-1])
            except (TokenBudgetError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                probe["question"] = ""
                rejections.append({**probe, "stage": "probe_generation", "reason": "invalid_probe", "error": str(exc)})
        dataset_store.put_bytes(keys["knowledge-units"], _jsonl([unit.to_dict() for unit in units]), media_type="application/x-ndjson")
        dataset_store.put_bytes(keys["probes"], _jsonl(probes), media_type="application/x-ndjson")
        checkpoint("discovery", [keys["knowledge-units"], keys["probes"]], len(probes))
    jobs.append_event(job_id, "stage_completed", {"stage": "discovery", "knowledge_units_created": len(units), "probes_created": len(probes)})

    responses = _read_jsonl(dataset_store, keys["student-responses"])
    jobs.append_event(job_id, "stage_started", {"stage": "student"})
    if not _completed_checkpoint(dataset_store, dataset_root, "student", checkpoint_identity, {}):
        student = StructuredAdapter(pool, _role_config(config, "student"))
        for probe in probes:
            if not probe.get("question"):
                continue
            try:
                raw = student.ask_budgeted(probe["question"], prompt_dir.joinpath("v2_student_probe.txt").read_text(encoding="utf-8"), context_limit=config.context_limit_tokens, role="student", prompt_version=config.prompt_versions.get("student_probe", "2.0"), evidence_ids=[], calls=calls)
                response_id = deterministic_id(probe["probe_id"], "student-response")
                responses.append({"student_response_id": response_id, "probe_id": probe["probe_id"], "knowledge_unit_id": probe["knowledge_unit_id"], "response": raw, "student_model": asdict(student.config), "model_role": "student", "prompt_version": config.prompt_versions.get("student_probe", "2.0"), "request_metadata": dict(calls[-1]), "source_asset_ids": probe["source_asset_ids"], "document_ids": probe["document_ids"], "evidence_ids": probe["evidence_ids"]})
            except (TokenBudgetError, RuntimeError) as exc:
                rejections.append({**probe, "stage": "student", "reason": normalized_error_category(exc), "error": str(exc)})
        dataset_store.put_bytes(keys["student-responses"], _jsonl(responses), media_type="application/x-ndjson")
        checkpoint("student", [keys["student-responses"]], len(responses))
    jobs.append_event(job_id, "stage_completed", {"stage": "student", "probes_completed": len(responses)})

    judgments = _read_jsonl(dataset_store, keys["probe-judgments"])
    selected = _read_jsonl(dataset_store, keys["selected-units"])
    candidates = _read_jsonl(dataset_store, keys["candidates"])
    jobs.append_event(job_id, "stage_started", {"stage": "judgment"})
    if not _completed_checkpoint(dataset_store, dataset_root, "judgment", checkpoint_identity, {}):
        judge = StructuredAdapter(pool, _role_config(config, "probe_judge"))
        qa = StructuredAdapter(pool, _role_config(config, "qa_generator"))
        units_by_id = {unit.knowledge_unit_id: unit for unit in units}
        probes_by_id = {probe["probe_id"]: probe for probe in probes}
        for response in responses:
            probe = probes_by_id[response["probe_id"]]
            unit = units_by_id[probe["knowledge_unit_id"]]
            cited = "\n".join(evidence_by_id[eid].text for eid in unit.source_evidence_ids if eid in evidence_by_id)
            try:
                raw = judge.ask_budgeted(prompt_dir.joinpath("v2_probe_judgment.txt").read_text(encoding="utf-8") + f"\nEVIDENCE:\n{cited}\nKNOWLEDGE UNIT:\n{unit.canonical_statement}\nSTUDENT RESPONSE:\n{response['response']}", "Return JSON with class (known, partial, unknown, invalid_probe) and reference_answer.", context_limit=config.context_limit_tokens, role="probe_judge", prompt_version=config.prompt_versions.get("probe_judgment", "2.0"), evidence_ids=list(unit.source_evidence_ids), calls=calls)
                data = _probed_json(raw, required=("class", "reference_answer"))
                probe_class = data["class"]
                if probe_class not in {"known", "partial", "unknown", "invalid_probe"}:
                    raise ValueError("Unsupported probe class")
                judgment_id = deterministic_id(probe["probe_id"], "judgment")
                judgment = {"probe_judgment_id": judgment_id, "probe_id": probe["probe_id"], "student_response_id": response["student_response_id"], "student_model": response["student_model"], "knowledge_unit_id": unit.knowledge_unit_id, "probe_class": probe_class, "reasons": data.get("reasons", {}), "reference_answer": data["reference_answer"], "model_role": "probe_judge", "model": asdict(judge.config), "prompt_version": config.prompt_versions.get("probe_judgment", "2.0"), "request_metadata": dict(calls[-1]), "source_asset_ids": probe["source_asset_ids"], "document_ids": probe["document_ids"], "evidence_ids": probe["evidence_ids"]}
                judgments.append(judgment)
                include = probe_class in config.include_classes or (probe_class == "known" and config.known_sample_rate > 0 and int(judgment_id[:8], 16) / 0xFFFFFFFF < config.known_sample_rate)
                if include and probe_class != "invalid_probe":
                    selected.append({**judgment, "selection_reason": "configured_probe_class"})
            except (TokenBudgetError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                rejections.append({**response, "stage": "judgment", "reason": normalized_error_category(exc), "error": str(exc)})
        dataset_store.put_bytes(keys["probe-judgments"], _jsonl(judgments), media_type="application/x-ndjson")
        dataset_store.put_bytes(keys["selected-units"], _jsonl(selected), media_type="application/x-ndjson")
        checkpoint("judgment", [keys["probe-judgments"]], len(judgments))
    jobs.append_event(job_id, "stage_completed", {"stage": "judgment", "probes_judged": len(judgments)})

    selected = _read_jsonl(dataset_store, keys["selected-units"])
    candidates = _read_jsonl(dataset_store, keys["candidates"])
    jobs.append_event(job_id, "stage_started", {"stage": "selection"})
    if not _completed_checkpoint(dataset_store, dataset_root, "selection", checkpoint_identity, {}):
        checkpoint("selection", [keys["selected-units"]], len(selected))
    jobs.append_event(job_id, "stage_completed", {"stage": "selection", "knowledge_units_selected": len(selected)})
    jobs.append_event(job_id, "stage_started", {"stage": "qa_generation"})
    if not _completed_checkpoint(dataset_store, dataset_root, "qa_generation", checkpoint_identity, {}):
        qa = StructuredAdapter(pool, _role_config(config, "qa_generator"))
        units_by_id = {unit.knowledge_unit_id: unit for unit in units}
        for selected_item in selected:
            unit = units_by_id[selected_item["knowledge_unit_id"]]
            cited = "\n".join(evidence_by_id[eid].text for eid in unit.source_evidence_ids if eid in evidence_by_id)
            try:
                qa_raw = qa.ask_budgeted(prompt_dir.joinpath("v2_probed_qa_generation.txt").read_text(encoding="utf-8") + f"\nEVIDENCE:\n{cited}\nKNOWLEDGE UNIT:\n{unit.canonical_statement}", "Return JSON with instruction and answer.", context_limit=config.context_limit_tokens, role="qa_generator", prompt_version=config.prompt_versions.get("probed_qa_generation", "2.0"), evidence_ids=list(unit.source_evidence_ids), calls=calls)
                qa_data = _probed_json(qa_raw, required=("instruction", "answer"))
                candidates.append({**selected_item, "instruction": qa_data["instruction"], "answer": qa_data["answer"], "model_role": "qa_generator", "model": asdict(qa.config), "prompt_version": config.prompt_versions.get("probed_qa_generation", "2.0"), "request_metadata": dict(calls[-1])})
            except (TokenBudgetError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                rejections.append({**selected_item, "stage": "qa_generation", "reason": normalized_error_category(exc), "error": str(exc)})
        dataset_store.put_bytes(keys["candidates"], _jsonl(candidates), media_type="application/x-ndjson")
        checkpoint("qa_generation", [keys["candidates"]], len(candidates))
    jobs.append_event(job_id, "stage_completed", {"stage": "qa_generation", "records_generated": len(candidates)})

    validations = _read_jsonl(dataset_store, keys["validations"])
    jobs.append_event(job_id, "stage_started", {"stage": "validation"})
    if not _completed_checkpoint(dataset_store, dataset_root, "validation", checkpoint_identity, {}):
        validator = StructuredAdapter(pool, _role_config(config, "validator"))
        for candidate in candidates:
            try:
                unit = next(unit for unit in units if unit.knowledge_unit_id == candidate["knowledge_unit_id"])
                cited = "\n".join(evidence_by_id[eid].text for eid in unit.source_evidence_ids if eid in evidence_by_id)
                judgment = {key: candidate[key] for key in ("probe_judgment_id", "probe_id", "student_response_id", "student_model", "knowledge_unit_id", "probe_class", "reasons", "reference_answer") if key in candidate}
                probe = next(probe for probe in probes if probe["probe_id"] == candidate["probe_id"])
                validation_material = f"{prompt_dir.joinpath('v2_probed_qa_validation.txt').read_text(encoding='utf-8')}\nORIGINAL EVIDENCE:\n{cited}\nKNOWLEDGE UNIT:\n{unit.canonical_statement}\nPROBE QUESTION:\n{probe['question']}\nSTUDENT RESPONSE:\n{next(item['response'] for item in responses if item['student_response_id'] == candidate['student_response_id'])}\nPROBE JUDGMENT:\n{json.dumps(judgment, sort_keys=True)}\nCANDIDATE:\n{json.dumps({'instruction': candidate['instruction'], 'answer': candidate['answer']}, sort_keys=True)}"
                raw = validator.ask_budgeted(validation_material, "Return JSON with decision accept or reject and reasons.", context_limit=config.context_limit_tokens, role="validator", prompt_version=config.prompt_versions.get("probed_qa_validation", "2.0"), evidence_ids=list(candidate["evidence_ids"]), calls=calls)
                data = _probed_json(raw, required=("decision",))
                if data["decision"] not in {"accept", "reject"}:
                    raise ValueError("Validator decision must be accept or reject")
                validations.append({**candidate, "decision": data["decision"], "reasons": data.get("reasons", {}), "validation_model": asdict(validator.config), "validation_prompt_version": config.prompt_versions.get("probed_qa_validation", "2.0"), "validation_request_metadata": dict(calls[-1])})
                if data["decision"] == "reject":
                    rejections.append({**candidate, "stage": "validation", "reason": "validation_rejected"})
            except (TokenBudgetError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                rejections.append({**candidate, "stage": "validation", "reason": normalized_error_category(exc), "error": str(exc)})
        dataset_store.put_bytes(keys["validations"], _jsonl(validations), media_type="application/x-ndjson")
        checkpoint("validation", [keys["validations"]], len(validations))
    jobs.append_event(job_id, "stage_completed", {"stage": "validation", "records_accepted": sum(item.get("decision") == "accept" for item in validations), "records_rejected": sum(item.get("decision") == "reject" for item in validations)})

    accepted = [item for item in validations if item.get("decision") == "accept"]
    records = [{"record_id": deterministic_id(dataset_id, item["probe_judgment_id"], item["instruction"], item["answer"]), "messages": [{"role": "user", "content": item["instruction"]}, {"role": "assistant", "content": item["answer"]}], "split": "", "metadata": {**{key: item.get(key, []) for key in ("source_asset_ids", "document_ids", "evidence_ids")}, "knowledge_unit_id": item["knowledge_unit_id"], "probe_id": item["probe_id"], "probe_class": item["probe_class"], "probe_judgment_id": item["probe_judgment_id"], "selection_reason": item.get("selection_reason", "configured_probe_class"), "student_model": item.get("student_model", asdict(_role_config(config, "student"))), "student_response_id": item.get("student_response_id"), "recipe": "knowledge-unit-probed-qa"}} for item in accepted]
    for record in records:
        cited = [
            evidence_by_id[item]
            for item in record["metadata"]["evidence_ids"]
            if item in evidence_by_id
        ]
        record["metadata"]["source_anchor_ids"] = _source_anchor_ids(cited)
        record["metadata"]["enrichment_id"] = _enrichment_id(
            cited,
            representation_type="training-record",
            generation_method="knowledge-unit-probed-qa",
            model_version=_role_config(config, "qa_generator").model,
            configuration=asdict(config),
        )
    records, duplicate_count = deduplicate_records(records, split_seed=config.split_seed)
    dataset_store.put_bytes(keys["records"], _jsonl(records), media_type="application/x-ndjson")
    dataset_store.put_bytes(keys["rejections"], _jsonl(rejections), media_type="application/x-ndjson")
    final_artifacts = {key: checksum(_jsonl(_read_jsonl(dataset_store, key)).decode()) for key in keys.values() if dataset_store.exists(key)}
    jobs.append_event(job_id, "stage_started", {"stage": "finalization"})
    _check_cancel(jobs, job_id, "finalization")
    payload = {**_identity_fields(identity), "dataset_name": dataset_name, "recipe": "knowledge-unit-probed-qa", "schema_version": "cognityx.dataforge.dataset/v1", "source_manifest_uri": input_manifest_uri, "input_selection_uri": dataset_store.uri(f"{identity.run_root}/input-selection.json"), "source_manifest_checksum": checksum(manifest), "configuration_checksum": checksum(asdict(config)), "effective_configuration": asdict(config), "models": {name: asdict(getattr(config, name) or config.generator) for name in ("knowledge_unit", "probe_generator", "student", "probe_judge", "qa_generator", "validator")}, "probing": {"probes_per_unit": config.probes_per_unit, "include_classes": list(config.include_classes), "known_sample_rate": config.known_sample_rate}, "prompt_versions": dict(config.prompt_versions), "split_seed": config.split_seed, "probe_count": len(probes), "judgment_count": len(judgments), "selected_count": len(selected), "accepted_count": len(records), "rejected_count": len(rejections), "duplicate_count": duplicate_count, "truncation_count": sum(item.get("status") == "truncated" for item in calls), "inference_failure_count": sum(item.get("status") == "failed" for item in calls), "train_count": sum(item["split"] == "train" for item in records), "validation_count": sum(item["split"] == "validation" for item in records), "test_count": sum(item["split"] == "test" for item in records), "records_uri": dataset_store.uri(keys["records"]), "records_checksum": final_artifacts[keys["records"]]}
    payload["provenance_refs"] = list(manifest.get("provenance_refs", ()))
    checkpoint("finalization", list(final_artifacts), len(records))
    dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", payload)
    jobs.append_event(job_id, "stage_completed", {"stage": "finalization", "records_published": len(records)})
    jobs.append_event(job_id, "build_completed", {"record_count": len(records), "dataset_id": dataset_id})
    jobs.set_state(job_id, "completed")
    return {**_identity_fields(identity), "recipe": "knowledge-unit-probed-qa", "record_count": len(records), "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json")}


def _build_knowledge_unit_staged(*, manifest: dict[str, Any], dataset_name: str, config: DataForgeConfig, runtime: StorageRuntime, jobs: JobRepository, inference_client: Any, dataset_store: Any, dataset_root: str, identity: BuildIdentity, input_manifest_uri: str) -> dict[str, Any]:
    job_id = identity.job_id
    dataset_id = identity.dataset_id
    dataset_version = identity.dataset_version
    prompt_dir = Path(__file__).with_name("prompts")
    knowledge_key = f"{dataset_root}/knowledge-units.jsonl"
    candidates_key = f"{dataset_root}/candidates.jsonl"
    validations_key = f"{dataset_root}/validations.jsonl"
    records_key = f"{dataset_root}/records.jsonl"
    rejections_key = f"{dataset_root}/rejections.jsonl"
    calls: list[dict[str, Any]] = []
    rejections = _read_jsonl(dataset_store, rejections_key)
    evidence_groups = []
    for ref in manifest["evidence_refs"]:
        evidence_store, evidence_key = resolve_storage_uri(runtime, ref)
        with evidence_store.open(evidence_key) as handle:
            evidence_groups.extend(load_evidence_jsonl(handle))
    evidence = combine_evidence([tuple(evidence_groups)])
    validate_context(manifest, evidence)
    evidence_availability(evidence)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    checkpoint_identity = {"dataset_name": dataset_name, **_identity_fields(identity), "source_manifest_checksum": checksum(manifest), "configuration_checksum": checksum(asdict(config)), "prompt_versions": dict(config.prompt_versions)}
    pool = inference_client if isinstance(inference_client, InferenceClientPool) else InferenceClientPool(config=config, injected_client=inference_client)
    pool.set_lineage(dataset_id=dataset_id, dataset_version=dataset_version, run_id=identity.run_id, job_id=job_id, recipe="knowledge-unit-qa", data_classification=config.data_classification)
    discovery = StructuredAdapter(pool, GeneratorConfig(**asdict(config.knowledge_unit or config.generator)))
    units = [KnowledgeUnit.from_dict(item) for item in _read_jsonl(dataset_store, knowledge_key)]
    jobs.append_event(job_id, "stage_started", {"stage": "discovery"})
    _check_cancel(jobs, job_id, "discovery")
    if not _completed_checkpoint(dataset_store, dataset_root, "discovery", checkpoint_identity, {}):
        for item in evidence:
            _check_cancel(jobs, job_id, "discovery")
            prompt = prompt_dir.joinpath("v1_knowledge_unit_discovery.txt").read_text(encoding="utf-8") + "\n\n" + item.text
            try:
                raw = discovery.ask_budgeted(prompt, "Return strict JSON for knowledge-unit discovery.", context_limit=config.context_limit_tokens, role="knowledge_unit", prompt_version=config.prompt_versions["knowledge_unit"], evidence_ids=[item.evidence_id], calls=calls)
                units.extend(parse_knowledge_units(raw, item.evidence_id, discovery.config.model, config.prompt_versions["knowledge_unit"]))
            except TokenBudgetError as exc:
                rejections.append({"stage": "discovery", "evidence_ids": [item.evidence_id], "reason": "token_budget_exceeded", "input_tokens": exc.input_tokens, "max_output_tokens": exc.max_output_tokens, "context_limit": exc.context_limit})
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                rejections.append({"stage": "discovery", "evidence_ids": [item.evidence_id], "reason": "malformed_model_json", "error": str(exc)})
        knowledge_bytes = _jsonl([unit.to_dict() for unit in units])
        dataset_store.put_bytes(knowledge_key, knowledge_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/model-calls-discovery.jsonl", _jsonl(calls), media_type="application/x-ndjson")
        _write_checkpoint(dataset_store, dataset_root, "discovery", checkpoint_identity, {knowledge_key: checksum(knowledge_bytes.decode())}, len(units))
    jobs.append_event(job_id, "stage_completed", {"stage": "discovery", "knowledge_units_created": len(units)})
    qa = StructuredAdapter(pool, GeneratorConfig(**asdict(config.qa_generator or config.generator)))
    candidates = _read_jsonl(dataset_store, candidates_key)
    jobs.append_event(job_id, "stage_started", {"stage": "generation"})
    _check_cancel(jobs, job_id, "generation")
    if not _completed_checkpoint(dataset_store, dataset_root, "generation", checkpoint_identity, {}):
        for unit in units:
            _check_cancel(jobs, job_id, "generation")
            cited_evidence = [evidence_by_id[item] for item in unit.source_evidence_ids if item in evidence_by_id]
            prompt = prompt_dir.joinpath("v1_knowledge_unit_generation.txt").read_text(encoding="utf-8") + "\n\nKNOWLEDGE UNIT:\n" + json.dumps(unit.to_dict(), sort_keys=True) + "\n\nCITED EVIDENCE:\n" + "\n".join(item.text for item in cited_evidence)
            try:
                raw = qa.ask_budgeted(prompt, "Return strict JSON with instruction and answer.", context_limit=config.context_limit_tokens, role="qa_generator", prompt_version=config.prompt_versions["generation"], evidence_ids=list(unit.source_evidence_ids), calls=calls)
                data = json.loads(raw)
                cited = [evidence_by_id[item] for item in unit.source_evidence_ids if item in evidence_by_id]
                candidates.append({"knowledge_unit_id": unit.knowledge_unit_id, "source_asset_ids": sorted({item.source_asset_id for item in cited if item.source_asset_id}), "document_ids": sorted({item.document_id for item in cited}), "evidence_ids": list(unit.source_evidence_ids), "instruction": str(data["instruction"]).strip(), "answer": str(data["answer"]).strip()})
            except TokenBudgetError as exc:
                rejections.append({"stage": "generation", "knowledge_unit_id": unit.knowledge_unit_id, "reason": "token_budget_exceeded", "input_tokens": exc.input_tokens, "max_output_tokens": exc.max_output_tokens, "context_limit": exc.context_limit})
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                rejections.append({"stage": "generation", "knowledge_unit_id": unit.knowledge_unit_id, "reason": "malformed_model_json", "error": str(exc)})
        candidates_bytes = _jsonl(candidates)
        dataset_store.put_bytes(candidates_key, candidates_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/model-calls-generation.jsonl", _jsonl(calls), media_type="application/x-ndjson")
        _write_checkpoint(dataset_store, dataset_root, "generation", checkpoint_identity, {candidates_key: checksum(candidates_bytes.decode())}, len(candidates))
    jobs.append_event(job_id, "stage_completed", {"stage": "generation", "records_generated": len(candidates)})
    validator = StructuredAdapter(pool, GeneratorConfig(**asdict(config.validator or config.generator)))
    validations = _read_jsonl(dataset_store, validations_key)
    jobs.append_event(job_id, "stage_started", {"stage": "validation"})
    _check_cancel(jobs, job_id, "validation")
    if not _completed_checkpoint(dataset_store, dataset_root, "validation", checkpoint_identity, {}):
        by_id = {unit.knowledge_unit_id: unit for unit in units}
        for candidate in candidates:
            _check_cancel(jobs, job_id, "validation")
            unit = by_id[candidate["knowledge_unit_id"]]
            cited_evidence = [evidence_by_id[item] for item in unit.source_evidence_ids if item in evidence_by_id]
            prompt = prompt_dir.joinpath("v1_knowledge_unit_validation.txt").read_text(encoding="utf-8") + "\n\nORIGINAL EVIDENCE:\n" + "\n".join(item.text for item in cited_evidence) + "\n\nKNOWLEDGE UNIT:\n" + json.dumps(unit.to_dict(), sort_keys=True) + "\n\nCANDIDATE:\n" + json.dumps(candidate, sort_keys=True)
            try:
                raw = validator.ask_budgeted(prompt, "Return accept or reject with reasons.", context_limit=config.context_limit_tokens, role="validator", prompt_version=config.prompt_versions["validation"], evidence_ids=list(unit.source_evidence_ids), calls=calls)
                data = json.loads(raw)
                decision = data.get("decision")
                if decision not in {"accept", "reject"}:
                    raise ValueError("Validator decision must be accept or reject")
                validations.append({"knowledge_unit_id": unit.knowledge_unit_id, "source_asset_ids": candidate.get("source_asset_ids", []), "document_ids": candidate.get("document_ids", []), "evidence_ids": list(unit.source_evidence_ids), "decision": decision, "reasons": data.get("reasons", {})})
                if decision == "reject":
                    rejections.append({**candidate, "stage": "validation", "reason": data.get("reasons", {})})
            except TokenBudgetError as exc:
                rejections.append({**candidate, "stage": "validation", "reason": "token_budget_exceeded", "input_tokens": exc.input_tokens, "max_output_tokens": exc.max_output_tokens, "context_limit": exc.context_limit})
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                rejections.append({**candidate, "stage": "validation", "reason": "malformed_model_json", "error": str(exc)})
        validations_bytes = _jsonl(validations)
        dataset_store.put_bytes(validations_key, validations_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/model-calls-validation.jsonl", _jsonl(calls), media_type="application/x-ndjson")
        _write_checkpoint(dataset_store, dataset_root, "validation", checkpoint_identity, {validations_key: checksum(validations_bytes.decode())}, len(validations))
    jobs.append_event(job_id, "stage_completed", {"stage": "validation", "records_accepted": sum(item.get("decision") == "accept" for item in validations), "records_rejected": sum(item.get("decision") == "reject" for item in validations)})
    accepted = {item["knowledge_unit_id"] for item in validations if item.get("decision") == "accept"}
    records = [{"record_id": deterministic_id(dataset_id, item["knowledge_unit_id"], item["instruction"], item["answer"]), "messages": [{"role": "user", "content": item["instruction"]}, {"role": "assistant", "content": item["answer"]}], "split": "", "metadata": {"recipe": "knowledge-unit-qa", "source_asset_ids": item.get("source_asset_ids", []), "document_ids": item.get("document_ids", []), "evidence_ids": item["evidence_ids"], "knowledge_unit_id": item["knowledge_unit_id"], "prompt_versions": dict(config.prompt_versions), "generator_model": (config.qa_generator or config.generator).model}} for item in candidates if item["knowledge_unit_id"] in accepted]
    for record in records:
        cited = [
            evidence_by_id[item]
            for item in record["metadata"]["evidence_ids"]
            if item in evidence_by_id
        ]
        record["metadata"]["source_anchor_ids"] = _source_anchor_ids(cited)
        record["metadata"]["enrichment_id"] = _enrichment_id(
            cited,
            representation_type="training-record",
            generation_method="knowledge-unit-qa",
            model_version=(config.qa_generator or config.generator).model,
            configuration=asdict(config),
        )
    records, duplicate_count = deduplicate_records(records, split_seed=config.split_seed)
    records_bytes = _jsonl(records)
    rejections_bytes = _jsonl(rejections)
    knowledge_bytes = _jsonl([unit.to_dict() for unit in units])
    validations_bytes = _jsonl(validations)
    dataset_store.put_bytes(records_key, records_bytes, media_type="application/x-ndjson")
    dataset_store.put_bytes(rejections_key, rejections_bytes, media_type="application/x-ndjson")
    final_artifacts = {knowledge_key: checksum(knowledge_bytes.decode()), candidates_key: checksum(_jsonl(candidates).decode()), validations_key: checksum(validations_bytes.decode()), records_key: checksum(records_bytes.decode()), rejections_key: checksum(rejections_bytes.decode())}
    jobs.append_event(job_id, "stage_started", {"stage": "finalization"})
    _check_cancel(jobs, job_id, "finalization")
    manifest_payload = {**_identity_fields(identity), "dataset_name": dataset_name, "recipe": "knowledge-unit-qa", "schema_version": "cognityx.dataforge.dataset/v1", "source_manifest_uri": input_manifest_uri, "input_selection_uri": dataset_store.uri(f"{identity.run_root}/input-selection.json"), "source_manifest_checksum": checksum(manifest), "configuration_checksum": checksum(asdict(config)), "effective_configuration": asdict(config), "models": {"knowledge_unit": asdict(config.knowledge_unit or config.generator), "qa_generator": asdict(config.qa_generator or config.generator), "validator": asdict(config.validator or config.generator)}, "prompt_versions": dict(config.prompt_versions), "split_seed": config.split_seed, "knowledge_unit_count": len(units), "candidate_count": len(candidates), "accepted_count": len(records), "rejected_count": len(rejections), "duplicate_count": duplicate_count, "truncation_count": sum(item.get("status") == "truncated" for item in calls), "inference_failure_count": sum(item.get("status") == "failed" for item in calls), "train_count": sum(item["split"] == "train" for item in records), "validation_count": sum(item["split"] == "validation" for item in records), "test_count": sum(item["split"] == "test" for item in records), "eval_count": sum(item["split"] != "train" for item in records), "validation_failure_counts": {"rejected": sum(item.get("decision") == "reject" for item in validations)}, "knowledge_units_uri": dataset_store.uri(knowledge_key), "knowledge_units_checksum": final_artifacts[knowledge_key], "validations_uri": dataset_store.uri(validations_key), "validations_checksum": final_artifacts[validations_key], "records_uri": dataset_store.uri(records_key), "records_checksum": final_artifacts[records_key], "model_calls_uri": dataset_store.uri(f"{dataset_root}/model-calls-validation.jsonl")}
    manifest_payload["provenance_refs"] = list(manifest.get("provenance_refs", ()))
    _write_checkpoint(dataset_store, dataset_root, "finalization", checkpoint_identity, final_artifacts, len(records))
    dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
    jobs.append_event(job_id, "stage_completed", {"stage": "finalization", "records_published": len(records)})
    jobs.append_event(job_id, "build_completed", {"record_count": len(records), "dataset_id": dataset_id})
    jobs.set_state(job_id, "completed")
    return {**_identity_fields(identity), "recipe": "knowledge-unit-qa", "record_count": len(records), "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json")}


def _evidence_provenance(item: Any, *, char_start: int, char_end: int) -> dict[str, Any]:
    return {
        "source_asset_ids": [item.source_asset_id] if item.source_asset_id else [],
        "document_ids": [item.document_id],
        "evidence_ids": [item.evidence_id],
        "source_anchor_ids": _source_anchor_ids([item]),
        "char_start": char_start,
        "char_end": char_end,
        "page_coordinates": {
            "physical_page_index": getattr(item, "physical_page_index", None),
            "page_number": getattr(item, "page_number", None),
            "printed_page_label": getattr(item, "printed_page_label", None),
            "pdf_page_label": getattr(item, "pdf_page_label", None),
        },
    }


def _paragraph_candidate(item: Any, *, dataset_name: str, char_start: int, char_end: int, text: str) -> dict[str, Any]:
    return {
        "candidate_id": deterministic_id(dataset_name, "paragraph-qa-candidate", item.evidence_id, str(char_start), str(char_end)),
        "evidence_id": item.evidence_id,
        "char_start": char_start,
        "char_end": char_end,
        "text": text,
        "source_text": text,
        "provenance": _evidence_provenance(item, char_start=char_start, char_end=char_end),
    }


def _build_paragraph_qualified(
    *,
    manifest: dict[str, Any],
    dataset_name: str,
    config: DataForgeConfig,
    runtime: StorageRuntime,
    jobs: JobRepository,
    inference_client: Any,
    dataset_store: Any,
    dataset_root: str,
    identity: BuildIdentity,
    input_manifest_uri: str,
) -> dict[str, Any]:
    recipe = "paragraph-qa-qualified"
    keys = {
        name: f"{dataset_root}/{name}.jsonl"
        for name in (
            "candidates",
            "answer-requirements",
            "source-answerability",
            "reference-qualification",
            "qualification-decisions",
            "accepted",
            "rejected",
            "needs-review",
            "records",
            "model-calls-generation",
            "model-calls",
        )
    }
    summary_key = f"{dataset_root}/qualification-summary.json"
    checkpoint_identity = {
        "dataset_name": dataset_name,
        "experiment_id": identity.experiment_id,
        "variant_id": identity.variant_id,
        "dataset_id": identity.dataset_id,
        "dataset_version": identity.dataset_version,
        "recipe": recipe,
        "source_manifest_checksum": checksum(manifest),
        "configuration_checksum": checksum(asdict(config)),
        "models": _qualified_models(config),
        "prompt_versions": dict(config.prompt_versions),
    }

    evidence_groups = []
    for ref in manifest["evidence_refs"]:
        store, key = resolve_storage_uri(runtime, ref)
        with store.open(key) as handle:
            evidence_groups.append(load_evidence_jsonl(handle))
    evidence = combine_evidence(evidence_groups)
    validate_context(manifest, evidence)
    evidence_availability(evidence)

    pool = inference_client if isinstance(inference_client, InferenceClientPool) else InferenceClientPool(config=config, injected_client=inference_client)
    pool.set_lineage(
        dataset_id=identity.dataset_id,
        dataset_version=identity.dataset_version,
        run_id=identity.run_id,
        job_id=identity.job_id,
        recipe=recipe,
        data_classification=config.data_classification,
    )
    calls = _read_jsonl(dataset_store, keys["model-calls"])
    if not calls:
        calls = _read_jsonl(dataset_store, keys["model-calls-generation"])
    rejections: list[dict[str, Any]] = []
    candidates = _read_jsonl(dataset_store, keys["candidates"])

    def checkpoint(stage: str, artifact_keys: list[str], rows: int) -> None:
        artifacts: dict[str, str] = {}
        for artifact_key in artifact_keys:
            with dataset_store.open(artifact_key) as handle:
                artifacts[artifact_key] = checksum(handle.read().decode("utf-8"))
        _write_checkpoint(dataset_store, dataset_root, stage, checkpoint_identity, artifacts, rows)

    _check_cancel(jobs, identity.job_id, "candidate_generation")
    jobs.append_event(identity.job_id, "stage_started", {"stage": "candidate_generation"})
    if not _completed_checkpoint(dataset_store, dataset_root, "candidate_generation", checkpoint_identity, {}):
        candidates = []
        generator = GeneratorAdapter(pool, GeneratorConfig(**asdict(config.generator)))
        prompt = Path(__file__).with_name("prompts").joinpath("v0_instruction_answer.txt").read_text(encoding="utf-8")
        for item in evidence:
            for start, end, text in paragraph_spans(item.text):
                _check_cancel(jobs, identity.job_id, "candidate_generation")
                candidate = _paragraph_candidate(
                    item,
                    dataset_name=dataset_name,
                    char_start=start,
                    char_end=end,
                    text=text,
                )
                candidate_id = candidate["candidate_id"]
                try:
                    generated = generator.generate_budgeted(
                        prompt + "\n\n" + text,
                        context_limit=config.context_limit_tokens,
                        role="generator",
                        prompt_version=config.prompt_versions["generation"],
                        evidence_ids=[item.evidence_id],
                        calls=calls,
                    )
                    candidate.update({
                        "status": "generated",
                        "question": generated["instruction"],
                        "reference": generated["answer"],
                        "request_metadata": dict(calls[-1]),
                    })
                except Exception as exc:  # noqa: BLE001 - record model failure
                    candidate.update({
                        "status": "generation_failed",
                        "failure_category": normalized_error_category(exc),
                        "error": str(exc),
                    })
                    rejections.append({
                        "candidate_id": candidate_id,
                        "stage": "candidate_generation",
                        "decision": "rejected",
                        "reason_codes": [normalized_error_category(exc)],
                        "error": str(exc),
                    })
                candidates.append(candidate)
        dataset_store.put_bytes(keys["candidates"], _jsonl(candidates), media_type="application/x-ndjson")
        dataset_store.put_bytes(keys["model-calls-generation"], _jsonl(calls), media_type="application/x-ndjson")
        checkpoint("candidate_generation", [keys["candidates"], keys["model-calls-generation"]], len(candidates))
    jobs.append_event(identity.job_id, "stage_completed", {
        "stage": "candidate_generation",
        "candidate_count": len(candidates),
        "generation_failure_count": sum(item.get("status") != "generated" for item in candidates),
    })

    answer_requirements = _read_jsonl(dataset_store, keys["answer-requirements"])
    source_answerability = _read_jsonl(dataset_store, keys["source-answerability"])
    reference_qualification = _read_jsonl(dataset_store, keys["reference-qualification"])
    decisions = _read_jsonl(dataset_store, keys["qualification-decisions"])
    accepted = _read_jsonl(dataset_store, keys["accepted"])
    needs_review = _read_jsonl(dataset_store, keys["needs-review"])
    records_payload = _read_jsonl(dataset_store, keys["records"])

    _check_cancel(jobs, identity.job_id, "qualification")
    jobs.append_event(identity.job_id, "stage_started", {"stage": "qualification"})
    if not _completed_checkpoint(dataset_store, dataset_root, "qualification", checkpoint_identity, {}):
        answer_requirements = []
        source_answerability = []
        reference_qualification = []
        decisions = []
        accepted = []
        needs_review = []
        qualification_rejections = [
            {
                "candidate_id": candidate["candidate_id"],
                "stage": "candidate_generation",
                "decision": "rejected",
                "reason_codes": [candidate.get("failure_category", "invalid_response")],
                "error": candidate.get("error", "generation failed"),
            }
            for candidate in candidates
            if candidate.get("status") != "generated"
        ]
        proposed_records: list[dict[str, Any]] = []
        pipeline = QualificationPipeline(
            pool=pool,
            role_configs={
                name: _role_config(config, name)
                for name in ("answer_requirements", "source_answerability", "reference_qualification")
            },
            context_limit=config.context_limit_tokens,
            prompt_versions=config.prompt_versions,
            max_attempts=config.qualification_max_attempts,
        )
        for candidate in candidates:
            if candidate.get("status") != "generated":
                continue
            _check_cancel(jobs, identity.job_id, "qualification")
            result_key = f"{dataset_root}/qualification-results/{candidate['candidate_id']}.json"
            if dataset_store.exists(result_key):
                with dataset_store.open(result_key) as handle:
                    frozen_result = json.load(handle)
                calls.extend(frozen_result.get("model_calls", []))
            else:
                call_start = len(calls)
                result = pipeline.qualify(
                    question=candidate["question"],
                    reference=candidate["reference"],
                    source_text=candidate["source_text"],
                    provenance=candidate["provenance"],
                    evidence_ids=[candidate["evidence_id"]],
                    calls=calls,
                )
                frozen_result = {
                    "candidate_id": candidate["candidate_id"],
                    "answer_requirements": result.answer_requirements,
                    "source_answerability": result.source_answerability,
                    "reference_qualification": result.reference_qualification,
                    "decision": result.decision,
                    "raw_attempts": result.raw_attempts,
                    "model_calls": calls[call_start:],
                }
                dataset_store.put_json_idempotent(result_key, frozen_result)
                jobs.append_event(identity.job_id, "qualification_record_completed", {
                    "candidate_id": candidate["candidate_id"],
                    "decision": result.decision["decision"],
                })
            common = {"candidate_id": candidate["candidate_id"]}
            if frozen_result["answer_requirements"] is not None:
                answer_requirements.append({**common, **frozen_result["answer_requirements"]})
            if frozen_result["source_answerability"] is not None:
                source_answerability.append({**common, **frozen_result["source_answerability"]})
            if frozen_result["reference_qualification"] is not None:
                reference_qualification.append({**common, **frozen_result["reference_qualification"]})
            decision = {**common, **frozen_result["decision"], "raw_attempts": frozen_result["raw_attempts"]}
            decisions.append(decision)
            if decision["decision"] == "accepted":
                accepted.append({**common, "question": candidate["question"], "reference": candidate["reference"]})
                provenance = candidate["provenance"]
                proposed_records.append(DatasetRecord(
                    record_id=deterministic_id(identity.dataset_id, candidate["candidate_id"], candidate["question"], candidate["reference"]),
                    messages=(
                        {"role": "user", "content": candidate["question"]},
                        {"role": "assistant", "content": candidate["reference"]},
                    ),
                    split="",
                    metadata={
                        "recipe": recipe,
                        **provenance,
                        "generator_model": config.generator.model,
                        "qualification_models": _qualified_models(config),
                        "prompt_versions": dict(config.prompt_versions),
                        "candidate_id": candidate["candidate_id"],
                        "enrichment_id": _enrichment_id(
                            [next(item for item in evidence if item.evidence_id == candidate["evidence_id"])],
                            representation_type="training-record",
                            generation_method=recipe,
                            model_version=config.generator.model,
                            configuration=asdict(config),
                        ),
                    },
                ).to_dict())
            elif decision["decision"] == "needs_review":
                needs_review.append({**common, "question": candidate["question"], "reference": candidate["reference"], **decision})
            else:
                qualification_rejections.append({**common, "stage": "qualification", "question": candidate["question"], "reference": candidate["reference"], **decision})

        records_payload, duplicate_count = deduplicate_records(proposed_records, split_seed=config.split_seed)
        for record in records_payload:
            metadata = dict(record["metadata"])
            if record["split"] == "train":
                metadata.update({"research_role": "training", "training_eligible": True})
            else:
                metadata.update({"research_role": f"legacy_{record['split']}", "training_eligible": False, "original_split": record["split"]})
            record["metadata"] = metadata

        artifact_rows = {
            keys["answer-requirements"]: answer_requirements,
            keys["source-answerability"]: source_answerability,
            keys["reference-qualification"]: reference_qualification,
            keys["qualification-decisions"]: decisions,
            keys["accepted"]: accepted,
            keys["rejected"]: qualification_rejections,
            keys["needs-review"]: needs_review,
            keys["records"]: records_payload,
            keys["model-calls"]: calls,
        }
        for artifact_key, rows in artifact_rows.items():
            dataset_store.put_bytes(artifact_key, _jsonl(rows), media_type="application/x-ndjson")
        rejections = qualification_rejections
        checkpoint("qualification", list(artifact_rows), len(decisions))
    else:
        rejections = _read_jsonl(dataset_store, keys["rejected"])
        duplicate_count = 0
    jobs.append_event(identity.job_id, "stage_completed", {
        "stage": "qualification",
        "accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "needs_review_count": len(needs_review),
    })

    reason_counts: dict[str, int] = {}
    for decision in decisions:
        for reason in decision.get("reason_codes", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = {
        "candidate_count": len(candidates),
        "accepted_count": len(records_payload),
        "qualification_accepted_count": len(accepted),
        "rejected_count": len(rejections),
        "qualification_rejected_count": sum(item.get("decision") == "rejected" for item in decisions),
        "needs_review_count": len(needs_review),
        "generation_failure_count": sum(item.get("status") != "generated" for item in candidates),
        "duplicate_count": duplicate_count,
        "reason_counts": reason_counts,
    }
    dataset_store.put_json_idempotent(summary_key, summary)

    final_artifacts: dict[str, str] = {}
    for artifact_key in [*keys.values(), summary_key]:
        with dataset_store.open(artifact_key) as handle:
            final_artifacts[artifact_key] = checksum(handle.read().decode("utf-8"))
    manifest_payload = {
        **_identity_fields(identity),
        "dataset_name": dataset_name,
        "recipe": recipe,
        "schema_version": "cognityx.dataforge.dataset/v1",
        "source_manifest_uri": input_manifest_uri,
        "input_selection_uri": dataset_store.uri(f"{identity.run_root}/input-selection.json"),
        "source_manifest_checksum": checksum(manifest),
        "configuration_checksum": checksum(asdict(config)),
        "effective_configuration": asdict(config),
        "models": _qualified_models(config),
        "prompt_versions": dict(config.prompt_versions),
        "split_seed": config.split_seed,
        **summary,
        "train_count": sum(record["split"] == "train" for record in records_payload),
        "validation_count": sum(record["split"] == "validation" for record in records_payload),
        "test_count": sum(record["split"] == "test" for record in records_payload),
        "eval_count": sum(record["split"] != "train" for record in records_payload),
        "records_uri": dataset_store.uri(keys["records"]),
        "records_checksum": final_artifacts[keys["records"]],
        "qualification_artifacts": {
            name: {"uri": dataset_store.uri(key), "checksum": final_artifacts[key]}
            for name, key in keys.items()
            if name != "records"
        },
        "qualification_summary_uri": dataset_store.uri(summary_key),
        "qualification_summary_checksum": final_artifacts[summary_key],
        "qualification_result_root_uri": dataset_store.uri(f"{dataset_root}/qualification-results"),
        "provenance_refs": list(manifest.get("provenance_refs", ())),
        "created_at": time.time(),
    }
    _check_cancel(jobs, identity.job_id, "finalization")
    _write_checkpoint(dataset_store, dataset_root, "finalization", checkpoint_identity, final_artifacts, len(records_payload))
    dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
    jobs.append_event(identity.job_id, "stage_completed", {"stage": "finalization", "records_published": len(records_payload)})
    jobs.append_event(identity.job_id, "build_completed", {"record_count": len(records_payload), "dataset_id": identity.dataset_id})
    jobs.set_state(identity.job_id, "completed")
    return {
        **_identity_fields(identity),
        "recipe": recipe,
        "record_count": len(records_payload),
        "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json"),
    }


def build_dataset(
    input_manifest_uri: str | None,
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
    source: str | None = None,
    experiment_id: str | None = None,
    requested_run_id: str | None = None,
    jobs_database: str | Path | None = None,
) -> dict[str, Any]:
    runtime = runtime or _runtime(storage_root, storage_config)
    jobs = jobs or (
        load_job_repository(jobs_database)
        if jobs_database is not None
        else JobRepository(":memory:")
    )
    config = DataForgeConfig.load(config_path)
    recipe = normalize_recipe(recipe, variant=variant)
    source_ref = source or input_manifest_uri
    if source_ref is None:
        raise ValueError("source is required")
    resolved_source = resolve_source(runtime, source_ref)
    manifest = resolved_source.source_manifest
    expected_source_checksum = resolved_source.checksum
    expected_config_checksum = checksum(asdict(config))
    selected_experiment_id = experiment_id or f"legacy-{deterministic_id(dataset_name)}"
    identity = BuildIdentity.create(
        experiment_id=selected_experiment_id,
        recipe=recipe,
        configuration_checksum=expected_config_checksum,
        source_checksum=expected_source_checksum,
        requested_run_id=requested_run_id,
    )
    if experiment_id is None:
        legacy_dataset_id = deterministic_id(dataset_name, recipe)
        legacy_dataset_version = deterministic_id(
            dataset_name,
            recipe,
            expected_source_checksum,
            expected_config_checksum,
            config.generator.model,
            config.validator.model if config.validator else "",
            *[f"{key}:{value}" for key, value in sorted(config.prompt_versions.items())],
        )
        identity = BuildIdentity(
            experiment_id=identity.experiment_id,
            variant_id=identity.variant_id,
            run_id=identity.run_id,
            job_id=identity.job_id,
            dataset_id=legacy_dataset_id,
            dataset_version=legacy_dataset_version,
        )
    dataset_store = runtime.for_role("dataset")
    dataset_root = (
        identity.dataset_root
        if experiment_id is not None
        else f"{identity.dataset_id}/{identity.dataset_version}"
    )
    manifest_key = f"{dataset_root}/manifest.json"
    if dataset_store.exists(manifest_key):
        with dataset_store.open(manifest_key) as handle:
            existing = json.load(handle)
        records_store, records_key = resolve_storage_uri(runtime, existing["records_uri"], role_name="dataset")
        with records_store.open(records_key) as handle:
            records_bytes = handle.read()
        model_identity_matches = (
            existing.get("models") == {"knowledge_unit": asdict(config.knowledge_unit or config.generator), "qa_generator": asdict(config.qa_generator or config.generator), "validator": asdict(config.validator or config.generator)}
            if recipe == "knowledge-unit-qa"
            else existing.get("models") == {name: asdict(getattr(config, name) or config.generator) for name in ("knowledge_unit", "probe_generator", "student", "probe_judge", "qa_generator", "validator")}
            if recipe == "knowledge-unit-probed-qa"
            else existing.get("models") == _qualified_models(config)
            if recipe == "paragraph-qa-qualified"
            else existing.get("generator") == asdict(config.generator)
        )
        if (
            existing.get("source_manifest_checksum") != expected_source_checksum
            or existing.get("configuration_checksum") != expected_config_checksum
            or (existing.get("recipe") or {"v0": "paragraph-qa", "v1": "knowledge-unit-qa"}.get(existing.get("variant"))) != recipe
            or not model_identity_matches
            or checksum(records_bytes.decode("utf-8")) != existing.get("records_checksum")
        ):
            raise ValueError("Existing immutable dataset does not match this build identity")
        if recipe == "knowledge-unit-qa":
            with dataset_store.open(_checkpoint_key(dataset_root, "finalization")) as handle:
                checkpoint_identity = json.load(handle)["identity"]
            if not all(_completed_checkpoint(dataset_store, dataset_root, stage, checkpoint_identity, {}) for stage in ("discovery", "generation", "validation", "finalization")):
                raise ValueError("Existing staged dataset is incomplete or has invalid checkpoints")
        if recipe == "knowledge-unit-probed-qa":
            with dataset_store.open(_checkpoint_key(dataset_root, "finalization")) as handle:
                checkpoint_identity = json.load(handle)["identity"]
            if not all(_completed_checkpoint(dataset_store, dataset_root, stage, checkpoint_identity, {}) for stage in ("discovery", "student", "judgment", "selection", "qa_generation", "validation", "finalization")):
                raise ValueError("Existing staged probed dataset is incomplete or has invalid checkpoints")
        if recipe == "paragraph-qa-qualified":
            with dataset_store.open(_checkpoint_key(dataset_root, "finalization")) as handle:
                checkpoint_identity = json.load(handle)["identity"]
            if not all(_completed_checkpoint(dataset_store, dataset_root, stage, checkpoint_identity, {}) for stage in ("candidate_generation", "qualification", "finalization")):
                raise ValueError("Existing qualified paragraph dataset is incomplete or has invalid checkpoints")
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
        identity.job_id,
        "dataforge.build",
        {
            **_identity_fields(identity),
            "dataset_name": dataset_name,
            "recipe": recipe,
            "source": source_ref,
            "source_manifest_uri": resolved_source.source_manifest_uri,
            "input_selection_uri": dataset_store.uri(f"{identity.run_root}/input-selection.json"),
        },
    )
    dataset_store.put_json_idempotent(
        f"{identity.run_root}/input-selection.json",
        {
            **resolved_source.selection_manifest,
            "experiment_id": identity.experiment_id,
            "variant_id": identity.variant_id,
            "run_id": identity.run_id,
            "configuration_checksum": expected_config_checksum,
        },
    )
    jobs.set_state(job.job_id, "running")
    jobs.append_event(job.job_id, "build_started", {
        **_identity_fields(identity),
        "source_run_id": manifest["run_id"],
    })
    jobs.append_event(job.job_id, "sources_resolved", {
        "source_count": len(manifest.get("source_assets", ())),
        "document_count": len(manifest.get("document_ids", ())),
        "evidence_ref_count": len(manifest["evidence_refs"]),
        "provenance_ref_count": len(resolved_source.provenance),
    })
    if recipe == "knowledge-unit-probed-qa":
        try:
            result = _build_knowledge_unit_probed(
                manifest=manifest, dataset_name=dataset_name, config=config, runtime=runtime, jobs=jobs,
                inference_client=InferenceClientPool(config=config, injected_client=inference_client) if inference_client is not None else InferenceClientPool(config=config),
                dataset_store=dataset_store, dataset_root=dataset_root, identity=identity,
                input_manifest_uri=resolved_source.source_manifest_uri,
            )
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            return result
        except Exception as exc:
            if jobs.get(job.job_id).state != "cancelled":
                jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
                jobs.set_state(job.job_id, "failed")
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            raise
    if recipe == "knowledge-unit-qa":
        try:
            result = _build_knowledge_unit_staged(
                manifest=manifest, dataset_name=dataset_name, config=config, runtime=runtime, jobs=jobs,
                inference_client=InferenceClientPool(config=config, injected_client=inference_client) if inference_client is not None else InferenceClientPool(config=config),
                dataset_store=dataset_store, dataset_root=dataset_root,
                identity=identity,
                input_manifest_uri=resolved_source.source_manifest_uri,
            )
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            return result
        except Exception as exc:
            if jobs.get(job.job_id).state != "cancelled":
                jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
                jobs.set_state(job.job_id, "failed")
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            raise
    if recipe == "paragraph-qa-qualified":
        try:
            result = _build_paragraph_qualified(
                manifest=manifest,
                dataset_name=dataset_name,
                config=config,
                runtime=runtime,
                jobs=jobs,
                inference_client=InferenceClientPool(config=config, injected_client=inference_client) if inference_client is not None else InferenceClientPool(config=config),
                dataset_store=dataset_store,
                dataset_root=dataset_root,
                identity=identity,
                input_manifest_uri=resolved_source.source_manifest_uri,
            )
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            return result
        except Exception as exc:
            if jobs.get(job.job_id).state != "cancelled":
                jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
                jobs.set_state(job.job_id, "failed")
            _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
            raise
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
        pool = InferenceClientPool(config=config, injected_client=inference_client)
        pool.set_lineage(
            dataset_id=identity.dataset_id,
            dataset_version=identity.dataset_version,
            run_id=identity.run_id,
            job_id=identity.job_id,
            recipe=recipe,
            data_classification=config.data_classification,
        )
        generator = GeneratorAdapter(pool, GeneratorConfig(**asdict(config.generator)))
        structured = StructuredAdapter(pool, GeneratorConfig(**asdict(config.generator)))
        validator_config = config.validator or config.generator
        validator = StructuredAdapter(pool, GeneratorConfig(**asdict(validator_config)))
        records: list[DatasetRecord] = []
        candidates: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        knowledge_units: list[KnowledgeUnit] = []
        calls: list[dict[str, Any]] = []
        prompt_dir = Path(__file__).with_name("prompts")
        prompt_template = prompt_dir.joinpath("v0_instruction_answer.txt").read_text(encoding="utf-8")
        jobs.append_event(job.job_id, "stage_started", {"stage": "generation"})
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
                candidate = _paragraph_candidate(
                    item,
                    dataset_name=dataset_name,
                    char_start=start,
                    char_end=end,
                    text=text,
                ) if recipe == "paragraph-qa" else {"evidence_id": item.evidence_id, "char_start": start, "char_end": end, "text": text}
                candidates.append(candidate)
                try:
                    if recipe == "knowledge-unit-qa":
                        generation_prompt = prompt_dir.joinpath("v1_knowledge_unit_generation.txt").read_text(encoding="utf-8")
                        generated = generator.generate(generation_prompt + "\n\n" + text)
                    else:
                        generated = generator.generate_budgeted(
                            prompt_template + "\n\n" + text,
                            context_limit=config.context_limit_tokens,
                            role="generator",
                            prompt_version=config.prompt_versions["generation"],
                            evidence_ids=[item.evidence_id],
                            calls=calls,
                        )
                        candidate.update({
                            "status": "generated",
                            "question": generated["instruction"],
                            "reference": generated["answer"],
                            "request_metadata": dict(calls[-1]),
                        })
                except Exception as exc:  # noqa: BLE001 - record model failure
                    candidate.update({"status": "generation_failed", "error": str(exc)})
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
                    split="",
                    metadata={
                        "recipe": recipe,
                        "source_asset_ids": [item.source_asset_id] if item.source_asset_id else [],
                        "document_ids": [item.document_id],
                        "evidence_ids": [item.evidence_id],
                        "source_anchor_ids": _source_anchor_ids([item]),
                        "enrichment_id": _enrichment_id(
                            [item],
                            representation_type="training-record",
                            generation_method=recipe,
                            model_version=config.generator.model,
                            configuration=asdict(config),
                        ),
                        "char_start": start,
                        "char_end": end,
                        "generator_model": config.generator.model,
                        "prompt_versions": dict(config.prompt_versions),
                        "request_metadata": dict(calls[-1]) if calls else {},
                        **({"knowledge_unit_id": unit.knowledge_unit_id} if unit is not None else {}),
                    },
                )
                records.append(record)
        jobs.append_event(job.job_id, "stage_completed", {
            "stage": "generation",
            "records_generated": len(records),
            "records_rejected": len(rejections),
        })
        records_payload, duplicate_count = deduplicate_records(
            [record.to_dict() for record in records],
            split_seed=config.split_seed,
        )
        candidates_bytes = _jsonl(candidates)
        records_bytes = _jsonl(records_payload)
        rejections_bytes = _jsonl(rejections)
        knowledge_units_bytes = _jsonl([unit.to_dict() for unit in knowledge_units])
        validations_bytes = _jsonl(validations)
        model_calls_bytes = _jsonl(calls)
        records_key = f"{dataset_root}/records.jsonl"
        manifest_payload = {
            **_identity_fields(identity),
            "dataset_name": dataset_name,
            "recipe": recipe,
            "source_manifest_uri": resolved_source.source_manifest_uri,
            "input_selection_uri": dataset_store.uri(f"{identity.run_root}/input-selection.json"),
            "source_manifest_checksum": expected_source_checksum,
            "provenance_refs": list(manifest.get("provenance_refs", ())),
            "provenance_checksums": resolved_source.selection_manifest.get(
                "provenance_checksums", {}
            ),
            "configuration_checksum": expected_config_checksum,
            "effective_configuration": asdict(config),
            "generator": asdict(config.generator),
            "schema_version": "cognityx.dataforge.dataset/v1",
            "prompt_versions": dict(config.prompt_versions),
            "split_seed": config.split_seed,
            "knowledge_unit_count": len(knowledge_units),
            "candidate_count": len(candidates),
            "accepted_count": len(records_payload),
            "rejected_count": len(rejections),
            "duplicate_count": duplicate_count,
            "train_count": sum(record["split"] == "train" for record in records_payload),
            "validation_count": sum(record["split"] == "validation" for record in records_payload),
            "test_count": sum(record["split"] == "test" for record in records_payload),
            "eval_count": sum(record["split"] != "train" for record in records_payload),
            "records_uri": dataset_store.uri(records_key),
            "records_checksum": checksum(records_bytes.decode("utf-8")),
            "knowledge_units_uri": dataset_store.uri(f"{dataset_root}/knowledge-units.jsonl"),
            "knowledge_units_checksum": checksum(knowledge_units_bytes.decode("utf-8")),
            "validations_uri": dataset_store.uri(f"{dataset_root}/validations.jsonl"),
            "validations_checksum": checksum(validations_bytes.decode("utf-8")),
            "validation_failure_counts": {"rejected": sum(item.get("decision") == "reject" for item in validations)},
            "model_calls_uri": dataset_store.uri(f"{dataset_root}/model-calls.jsonl"),
            "inference_failure_count": sum(item.get("status") == "failed" for item in calls),
            "truncation_count": sum(item.get("status") == "truncated" for item in calls),
            "created_at": time.time(),
        }
        dataset_store.put_bytes(records_key, records_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/candidates.jsonl", candidates_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/rejections.jsonl", rejections_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/knowledge-units.jsonl", knowledge_units_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/validations.jsonl", validations_bytes, media_type="application/x-ndjson")
        dataset_store.put_bytes(f"{dataset_root}/model-calls.jsonl", model_calls_bytes, media_type="application/x-ndjson")
        _check_cancel(jobs, job.job_id, "publication")
        jobs.append_event(job.job_id, "stage_started", {"stage": "publication"})
        dataset_store.put_json_idempotent(f"{dataset_root}/manifest.json", manifest_payload)
        jobs.append_event(job.job_id, "stage_completed", {"stage": "publication", "records_published": len(records_payload)})
        jobs.append_event(job.job_id, "build_completed", {"record_count": len(records_payload), "dataset_id": identity.dataset_id})
        jobs.set_state(job.job_id, "completed")
        _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
        return {**_identity_fields(identity), "recipe": recipe, "record_count": len(records_payload), "dataset_manifest_uri": dataset_store.uri(f"{dataset_root}/manifest.json")}
    except KeyboardInterrupt:
        jobs.append_event(job.job_id, "build_cancelled", {})
        jobs.set_state(job.job_id, "cancelled")
        _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
        raise RuntimeError("DataForge build cancelled")
    except Exception as exc:
        if jobs.get(job.job_id).state != "cancelled":
            jobs.append_event(job.job_id, "build_failed", {"error": str(exc), "error_type": type(exc).__name__})
            jobs.set_state(job.job_id, "failed")
        _persist_run_events(dataset_store, identity.run_root, jobs, job.job_id)
        raise
