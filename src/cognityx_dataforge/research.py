from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from cognityx_storage import StorageRuntime

from cognityx_dataforge.dataset import checksum, deterministic_id
from cognityx_dataforge.source import resolve_storage_uri


EVALUATION_SET_SCHEMA = "cognityx.dataforge.evaluation-set/v1"
RESEARCH_PACKAGE_SCHEMA = "cognityx.dataforge.research-package/v1"
EVALUATION_ROLES = frozenset({
    "exact_recall",
    "paraphrase_evaluation",
    "heldout_knowledge_unit",
})


def _jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


def _put_bytes_idempotent(store: Any, key: str, content: bytes, *, media_type: str) -> None:
    if store.exists(key):
        with store.open(key) as handle:
            if handle.read() != content:
                raise ValueError(f"Existing immutable artifact does not match: {key}")
        return
    store.put_bytes(key, content, media_type=media_type)


def _load_json(runtime: StorageRuntime, uri: str, *, role: str = "dataset") -> dict[str, Any]:
    store, key = resolve_storage_uri(runtime, uri, role_name=role)
    with store.open(key) as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {uri}")
    return value


def _load_jsonl_uri(runtime: StorageRuntime, uri: str, *, role: str = "dataset") -> tuple[list[dict[str, Any]], bytes]:
    store, key = resolve_storage_uri(runtime, uri, role_name=role)
    with store.open(key) as handle:
        raw = handle.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {uri}")
    return rows, raw


def _validate_evaluation_record(record: dict[str, Any], role: str) -> dict[str, Any]:
    if not record.get("record_id"):
        raise ValueError("Evaluation records require record_id")
    existing_role = record.get("research_role") or record.get("metadata", {}).get("research_role")
    if existing_role and existing_role != role:
        raise ValueError(f"Evaluation record {record['record_id']} has research_role {existing_role!r}, expected {role!r}")
    eligible = record.get("training_eligible", record.get("metadata", {}).get("training_eligible", False))
    if eligible is not False:
        raise ValueError("evaluation_record_marked_trainable")
    if role in {"paraphrase_evaluation", "heldout_knowledge_unit"}:
        group_id = record.get("fact_group_id") or record.get("knowledge_unit_id")
        if not group_id:
            raise ValueError(f"Evaluation record {record['record_id']} needs fact_group_id or knowledge_unit_id")
        provenance = record.get("record_provenance") or record.get("source_evidence")
        if not provenance:
            raise ValueError(f"Evaluation record {record['record_id']} needs record_provenance or source_evidence")
    value = dict(record)
    value["split"] = "evaluation"
    value["research_role"] = role
    value["training_eligible"] = False
    metadata = dict(value.get("metadata", {}))
    metadata.update({"research_role": role, "training_eligible": False})
    value["metadata"] = metadata
    return value


def freeze_evaluation_set(
    runtime: StorageRuntime,
    *,
    evaluation_set_name: str,
    research_role: str,
    records: Iterable[dict[str, Any]],
    source_refs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if research_role not in EVALUATION_ROLES:
        raise ValueError(f"Unsupported evaluation research_role: {research_role}")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        value = _validate_evaluation_record(record, research_role)
        record_id = str(value["record_id"])
        if record_id in seen:
            raise ValueError(f"Duplicate evaluation record id: {record_id}")
        seen.add(record_id)
        validated.append(value)
    if not validated:
        raise ValueError("An evaluation set cannot be empty")
    records_bytes = _jsonl(validated)
    records_checksum = checksum(records_bytes.decode("utf-8"))
    normalized_source_refs = source_refs or []
    evaluation_set_id = deterministic_id(evaluation_set_name, research_role)
    evaluation_set_version = deterministic_id(
        evaluation_set_id,
        records_checksum,
        checksum(normalized_source_refs),
    )
    root = f"dataforge/evaluation-sets/{evaluation_set_id}/{evaluation_set_version}"
    store = runtime.for_role("dataset")
    records_key = f"{root}/records.jsonl"
    manifest_key = f"{root}/manifest.json"
    manifest = {
        "schema": EVALUATION_SET_SCHEMA,
        "evaluation_set_id": evaluation_set_id,
        "evaluation_set_version": evaluation_set_version,
        "evaluation_set_name": evaluation_set_name,
        "research_role": research_role,
        "training_eligible": False,
        "record_count": len(validated),
        "records_uri": store.uri(records_key),
        "records_checksum": records_checksum,
        "source_refs": normalized_source_refs,
    }
    _put_bytes_idempotent(store, records_key, records_bytes, media_type="application/x-ndjson")
    store.put_json_idempotent(manifest_key, manifest)
    return {**manifest, "manifest_uri": store.uri(manifest_key)}


def create_exact_recall_set(
    runtime: StorageRuntime,
    dataset_manifest_uri: str,
    *,
    evaluation_set_name: str | None = None,
) -> dict[str, Any]:
    manifest = _load_json(runtime, dataset_manifest_uri)
    if manifest.get("schema_version") != "cognityx.dataforge.dataset/v1":
        raise ValueError("Exact-recall source must be a DataForge dataset v1 manifest")
    records, raw = _load_jsonl_uri(runtime, manifest["records_uri"])
    if checksum(raw.decode("utf-8")) != manifest.get("records_checksum"):
        raise ValueError("Dataset records checksum verification failed")
    copied: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("metadata", {})
        role = metadata.get("research_role")
        if record.get("split") != "train" or role not in {None, "training"}:
            continue
        source_record_id = str(record["record_id"])
        copied.append({
            **record,
            "record_id": deterministic_id(source_record_id, "exact-recall", manifest["dataset_version"]),
            "source_record_id": source_record_id,
            "split": "evaluation",
            "research_role": "exact_recall",
            "training_eligible": False,
            "metadata": {
                **metadata,
                "source_record_id": source_record_id,
                "research_role": "exact_recall",
                "training_eligible": False,
            },
        })
    return freeze_evaluation_set(
        runtime,
        evaluation_set_name=evaluation_set_name or f"{manifest['dataset_name']}-exact-recall",
        research_role="exact_recall",
        records=copied,
        source_refs=[{
            "dataset_manifest_uri": dataset_manifest_uri,
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "records_checksum": manifest["records_checksum"],
        }],
    )


def import_evaluation_set(
    runtime: StorageRuntime,
    input_path: str | Path,
    *,
    evaluation_set_name: str,
    research_role: str,
) -> dict[str, Any]:
    path = Path(input_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return freeze_evaluation_set(
        runtime,
        evaluation_set_name=evaluation_set_name,
        research_role=research_role,
        records=rows,
        source_refs=[{"input_path": str(path), "input_checksum": checksum(path.read_text(encoding="utf-8"))}],
    )


def create_research_package(
    runtime: StorageRuntime,
    *,
    package_name: str,
    dataset_manifest_uri: str,
    evaluation_manifest_uris: Iterable[str],
) -> dict[str, Any]:
    dataset = _load_json(runtime, dataset_manifest_uri)
    if dataset.get("schema_version") != "cognityx.dataforge.dataset/v1":
        raise ValueError("Research package dataset must use cognityx.dataforge.dataset/v1")
    records, raw = _load_jsonl_uri(runtime, dataset["records_uri"])
    if checksum(raw.decode("utf-8")) != dataset.get("records_checksum"):
        raise ValueError("Dataset records checksum verification failed")
    if len(records) != dataset.get("accepted_count"):
        raise ValueError("Dataset accepted_count does not match its records")

    evaluation_sets: list[dict[str, Any]] = []
    roles: set[str] = set()
    for uri in evaluation_manifest_uris:
        manifest = _load_json(runtime, uri)
        if manifest.get("schema") != EVALUATION_SET_SCHEMA:
            raise ValueError(f"Unsupported evaluation manifest schema at {uri}")
        role = str(manifest.get("research_role"))
        if role in roles:
            raise ValueError(f"Duplicate evaluation research_role: {role}")
        roles.add(role)
        rows, evaluation_raw = _load_jsonl_uri(runtime, manifest["records_uri"])
        if checksum(evaluation_raw.decode("utf-8")) != manifest.get("records_checksum"):
            raise ValueError(f"Evaluation records checksum verification failed for {role}")
        if len(rows) != manifest.get("record_count"):
            raise ValueError(f"Evaluation record_count mismatch for {role}")
        evaluation_sets.append({
            "research_role": role,
            "manifest_uri": uri,
            "evaluation_set_id": manifest["evaluation_set_id"],
            "evaluation_set_version": manifest["evaluation_set_version"],
            "records_checksum": manifest["records_checksum"],
            "record_count": manifest["record_count"],
        })
    if "exact_recall" not in roles:
        raise ValueError("Research package requires an exact_recall evaluation set")

    identity = {
        "dataset": {
            "manifest_uri": dataset_manifest_uri,
            "dataset_id": dataset["dataset_id"],
            "dataset_version": dataset["dataset_version"],
            "records_checksum": dataset["records_checksum"],
        },
        "evaluation_sets": sorted(
            evaluation_sets,
            key=lambda item: item["research_role"],
        ),
    }
    package_id = deterministic_id(package_name)
    package_version = deterministic_id(package_id, checksum(identity))
    root = f"dataforge/research-packages/{package_id}/{package_version}"
    store = runtime.for_role("dataset")
    manifest_key = f"{root}/manifest.json"
    package = {
        "schema": RESEARCH_PACKAGE_SCHEMA,
        "research_package_id": package_id,
        "research_package_version": package_version,
        "package_name": package_name,
        "dataset": {
            "manifest_uri": dataset_manifest_uri,
            "dataset_id": dataset["dataset_id"],
            "dataset_version": dataset["dataset_version"],
            "records_checksum": dataset["records_checksum"],
            "record_count": dataset["accepted_count"],
        },
        "evaluation_sets": sorted(evaluation_sets, key=lambda item: item["research_role"]),
    }
    store.put_json_idempotent(manifest_key, package)
    return {**package, "manifest_uri": store.uri(manifest_key)}


def load_research_package(runtime: StorageRuntime, manifest_uri: str) -> dict[str, Any]:
    package = _load_json(runtime, manifest_uri)
    if package.get("schema") != RESEARCH_PACKAGE_SCHEMA:
        raise ValueError("Unsupported research package schema")
    return package
