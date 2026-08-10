from __future__ import annotations

import json
from pathlib import Path

import pytest
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.dataset import checksum
from cognityx_dataforge.research import (
    create_exact_recall_set,
    create_research_package,
    freeze_evaluation_set,
    import_evaluation_set,
)
from cognityx_dataforge.source import resolve_storage_uri


PACK = Path(__file__).parents[1] / "design_input" / "ift_research_foundation_v1"


def _dataset(runtime: StorageRuntime) -> tuple[str, list[dict]]:
    store = runtime.for_role("dataset")
    records = [
        {
            "record_id": "training-record-1",
            "messages": [
                {"role": "user", "content": "What must a manager state?"},
                {"role": "assistant", "content": "The actual documented reason."},
            ],
            "split": "train",
            "metadata": {
                "research_role": "training",
                "training_eligible": True,
                "document_ids": ["document-1"],
                "evidence_ids": ["evidence-1"],
            },
        },
        {
            "record_id": "legacy-validation-1",
            "messages": [
                {"role": "user", "content": "Legacy question"},
                {"role": "assistant", "content": "Legacy answer"},
            ],
            "split": "validation",
            "metadata": {"research_role": "legacy_validation", "training_eligible": False},
        },
    ]
    raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in records)
    records_uri = store.put_bytes("fixture/dataset/records.jsonl", raw, media_type="application/x-ndjson").uri
    manifest = {
        "schema_version": "cognityx.dataforge.dataset/v1",
        "dataset_id": "dataset-1",
        "dataset_version": "version-1",
        "dataset_name": "fixture",
        "accepted_count": 2,
        "records_uri": records_uri,
        "records_checksum": checksum(raw.decode("utf-8")),
    }
    return store.put_json("fixture/dataset/manifest.json", manifest).uri, records


def test_freeze_exact_recall_import_and_research_package(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    dataset_uri, source_records = _dataset(runtime)
    exact = create_exact_recall_set(runtime, dataset_uri)
    assert exact["record_count"] == 1
    assert exact["training_eligible"] is False
    exact_store, exact_key = resolve_storage_uri(runtime, exact["records_uri"], role_name="dataset")
    with exact_store.open(exact_key) as handle:
        exact_record = json.loads(handle.readline())
    assert exact_record["record_id"] != source_records[0]["record_id"]
    assert exact_record["source_record_id"] == source_records[0]["record_id"]
    assert exact_record["research_role"] == "exact_recall"
    assert exact_record["split"] == "evaluation"

    paraphrase = import_evaluation_set(
        runtime,
        PACK / "empirical/evaluation_sets/paraphrase_import_fixture_v1.jsonl",
        evaluation_set_name="paraphrase-import-v1",
        research_role="paraphrase_evaluation",
    )
    assert paraphrase["record_count"] == 4
    repeated = import_evaluation_set(
        runtime,
        PACK / "empirical/evaluation_sets/paraphrase_import_fixture_v1.jsonl",
        evaluation_set_name="paraphrase-import-v1",
        research_role="paraphrase_evaluation",
    )
    assert repeated["evaluation_set_version"] == paraphrase["evaluation_set_version"]

    package = create_research_package(
        runtime,
        package_name="qualification-comparison-v1",
        dataset_manifest_uri=dataset_uri,
        evaluation_manifest_uris=[exact["manifest_uri"], paraphrase["manifest_uri"]],
    )
    assert package["schema"] == "cognityx.dataforge.research-package/v1"
    assert {item["research_role"] for item in package["evaluation_sets"]} == {
        "exact_recall", "paraphrase_evaluation",
    }
    repeated_package = create_research_package(
        runtime,
        package_name="qualification-comparison-v1",
        dataset_manifest_uri=dataset_uri,
        evaluation_manifest_uris=[exact["manifest_uri"], paraphrase["manifest_uri"]],
    )
    assert repeated_package["research_package_version"] == package["research_package_version"]


def test_evaluation_freeze_rejects_leakage_duplicates_and_missing_provenance(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    valid = {
        "record_id": "one",
        "research_role": "paraphrase_evaluation",
        "training_eligible": False,
        "fact_group_id": "fact-1",
        "question": "Question",
        "gold_reference": "Answer",
        "record_provenance": {"evidence_ids": ["evidence-1"]},
    }
    with pytest.raises(ValueError, match="Duplicate evaluation record id"):
        freeze_evaluation_set(
            runtime,
            evaluation_set_name="duplicates",
            research_role="paraphrase_evaluation",
            records=[valid, valid],
        )
    with pytest.raises(ValueError, match="evaluation_record_marked_trainable"):
        freeze_evaluation_set(
            runtime,
            evaluation_set_name="leakage",
            research_role="paraphrase_evaluation",
            records=[{**valid, "training_eligible": True}],
        )
    with pytest.raises(ValueError, match="record_provenance"):
        freeze_evaluation_set(
            runtime,
            evaluation_set_name="missing-provenance",
            research_role="paraphrase_evaluation",
            records=[{key: value for key, value in valid.items() if key != "record_provenance"}],
        )
