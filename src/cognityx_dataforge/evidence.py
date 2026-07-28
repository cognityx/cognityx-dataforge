from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from cognityx_ingest.models import Evidence


def load_run_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "cognityx.ingest.run-manifest/v1":
        raise ValueError("Unsupported run manifest schema")
    return payload


def load_evidence_records(payload: dict[str, Any]) -> tuple[Evidence, ...]:
    records = payload.get("evidence") or ()
    return tuple(Evidence.from_dict(item) for item in records)


def validate_context(manifest: dict[str, Any], evidence: tuple[Evidence, ...]) -> None:
    expected_document = manifest.get("document_id")
    ids = {item.evidence_id for item in evidence}
    for ref in manifest.get("evidence_ids", []):
        if ref not in ids:
            raise ValueError(f"Missing referenced evidence: {ref}")
    if expected_document and any(item.document_id != expected_document for item in evidence):
        raise ValueError("Manifest evidence is not context-consistent")


def evidence_availability(evidence: tuple[Evidence, ...]) -> None:
    for item in evidence:
        if not item.text.strip():
            raise ValueError(f"Empty evidence: {item.evidence_id}")

