from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from cognityx_ingest.models import Evidence


RUN_SCHEMA = "cognityx.ingest.run"


def load_run_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != RUN_SCHEMA:
        raise ValueError("Unsupported run manifest schema")
    if not payload.get("run_id") or not payload.get("context_id"):
        raise ValueError("Ingest run manifest requires run_id and context_id")
    if not isinstance(payload.get("evidence_refs"), list):
        raise ValueError("Ingest run manifest requires evidence_refs")
    return payload


def load_evidence_jsonl(handle: Iterable[bytes | str]) -> tuple[Evidence, ...]:
    records: list[Evidence] = []
    for line_number, line in enumerate(handle, 1):
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.strip():
            continue
        try:
            records.append(Evidence.from_dict(json.loads(text)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid evidence JSONL at line {line_number}") from exc
    return tuple(records)


def combine_evidence(groups: Iterable[tuple[Evidence, ...]]) -> tuple[Evidence, ...]:
    evidence = [item for group in groups for item in group]
    evidence.sort(key=lambda item: (
        item.document_id,
        item.sequence_number if item.sequence_number is not None else 2**63,
        item.page_number,
        item.char_start,
        item.evidence_id,
    ))
    ids: set[str] = set()
    for item in evidence:
        if item.evidence_id in ids:
            raise ValueError(f"Duplicate evidence id: {item.evidence_id}")
        ids.add(item.evidence_id)
    return tuple(evidence)


def validate_context(manifest: Mapping[str, Any], evidence: tuple[Evidence, ...]) -> None:
    run_id = manifest["run_id"]
    context_id = manifest["context_id"]
    document_ids = set(manifest.get("document_ids", []))
    assets = {item["asset_id"]: item for item in manifest.get("source_assets", [])}
    expected_assets = set(assets)
    for item in evidence:
        if item.run_id != run_id:
            raise ValueError(f"Evidence {item.evidence_id} has inconsistent run_id")
        if item.context_id != context_id:
            raise ValueError(f"Evidence {item.evidence_id} has inconsistent context_id")
        if document_ids and item.document_id not in document_ids:
            raise ValueError(f"Evidence {item.evidence_id} references an unknown document")
        if item.source_asset_id and expected_assets and item.source_asset_id not in expected_assets:
            raise ValueError(f"Evidence {item.evidence_id} references an unknown SourceAsset")
        asset = assets.get(item.source_asset_id)
        if asset and item.source_sha256 and item.source_sha256 != asset.get("sha256"):
            raise ValueError(f"Evidence {item.evidence_id} has inconsistent SourceAsset lineage")
    if not evidence and manifest.get("evidence_refs"):
        raise ValueError("Ingest run references evidence but no records were found")


def evidence_availability(evidence: tuple[Evidence, ...]) -> None:
    for item in evidence:
        if not item.text.strip():
            raise ValueError(f"Empty evidence: {item.evidence_id}")
