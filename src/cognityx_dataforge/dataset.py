from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def checksum(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def deterministic_id(*parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:32]


def split_for_index(index: int) -> str:
    return "train" if index % 10 else "eval"


def split_for_group(group_id: str, seed: str = "dataforge-v1") -> str:
    bucket = int(hashlib.sha256(f"{seed}\x1f{group_id}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def record_group(metadata: dict[str, Any]) -> str:
    for key in ("source_asset_ids", "document_ids"):
        values = metadata.get(key) or ()
        if values:
            return f"{key}:{sorted(str(item) for item in values)[0]}"
    if metadata.get("knowledge_unit_id"):
        return f"knowledge_unit:{metadata['knowledge_unit_id']}"
    evidence = metadata.get("evidence_ids") or ()
    if evidence:
        return f"evidence:{sorted(str(item) for item in evidence)[0]}"
    return "ungrouped"


def deduplicate_records(
    records: list[dict[str, Any]],
    *,
    split_seed: str,
) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        messages = record.get("messages", ())
        fingerprint = checksum(messages)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        value = dict(record)
        metadata = dict(value.get("metadata", {}))
        group_id = record_group(metadata)
        metadata["split_group_id"] = group_id
        value["metadata"] = metadata
        value["split"] = split_for_group(group_id, split_seed)
        unique.append(value)
    return unique, duplicates
