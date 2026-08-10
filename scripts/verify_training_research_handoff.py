"""Run the DataForge-to-Training research handoff without a model or GPU."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
TRAINING_REPOSITORY = REPOSITORY.parent / "cognityx-training"
TRAINING_SOURCE = TRAINING_REPOSITORY / "src"
training_sites = sorted((TRAINING_REPOSITORY / ".venv/lib").glob("python*/site-packages"))
if not TRAINING_SOURCE.is_dir() or not training_sites:
    raise SystemExit(f"Sibling Training checkout and environment are required: {TRAINING_REPOSITORY}")
sys.path.append(str(training_sites[-1]))

from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset
from cognityx_dataforge.research import (
    create_exact_recall_set,
    create_research_package,
    import_evaluation_set,
)
from cognityx_dataforge.source import resolve_storage_uri


sys.path.insert(0, str(TRAINING_SOURCE))

from cognityx_training.dataset_pipeline import DataForgeDatasetReader  # noqa: E402
from cognityx_training.lineage import build_lineage_ids  # noqa: E402
from cognityx_training.publication import TrainingPublisher, prediction_rows  # noqa: E402
from cognityx_training.tracking import NoOpTracker, completed_run_payload  # noqa: E402


class FixtureInferenceClient:
    def chat(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Analyze only the question" in prompt:
            value = {"required_slots": ["documented_reason"], "answer_structure": "short_factual_rule"}
        elif "Use the frozen question requirements" in prompt:
            value = {"answerable_at_requested_specificity": True, "slot_values": {"documented_reason": "actual documented reason for a decision"}, "missing_slots": []}
        elif "Compare the generated reference" in prompt:
            value = {"answers_question": True, "required_slot_coverage": 1.0, "unsupported_claims": [], "contradicted_claims": [], "premise_restatement": False}
        else:
            value = {"instruction": "Under the Clear Moon Principle, what must a manager state?", "answer": "The actual documented reason for a decision."}
        return {"choices": [{"message": {"content": json.dumps(value)}}]}

    def count_input_tokens(self, **kwargs):
        return 20


def _read_json(runtime: StorageRuntime, uri: str) -> dict:
    store, key = resolve_storage_uri(runtime, uri, role_name="dataset")
    with store.open(key) as source:
        return json.load(source)


def _read_jsonl(runtime: StorageRuntime, uri: str) -> list[dict]:
    store, key = resolve_storage_uri(runtime, uri, role_name="dataset")
    with store.open(key) as source:
        return [json.loads(line) for line in source.read().decode().splitlines() if line]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cognityx-research-handoff-") as temporary:
        root = Path(temporary)
        runtime = StorageRuntime.from_config(StorageConfig.built_in(root=root / "storage"))
        artifact = runtime.for_role("artifact")
        evidence = {
            "evidence_id": "evidence-clear-moon",
            "document_id": "document-1",
            "source_asset_id": "asset-1",
            "page_number": 3,
            "physical_page_index": 2,
            "printed_page_label": "1",
            "text": "PR-01 Clear Moon Principle. A manager must state the actual documented reason for a decision.",
            "char_start": 0,
            "char_end": 94,
            "source_sha256": "sha-1",
            "context_id": "context-1",
            "run_id": "ingest-run-research",
        }
        evidence_uri = artifact.put_bytes(
            "integration/evidence.jsonl",
            (json.dumps(evidence) + "\n").encode(),
            media_type="application/x-ndjson",
        ).uri
        source_uri = artifact.put_json("integration/manifest.json", {
            "schema": "cognityx.ingest.run",
            "run_id": "ingest-run-research",
            "context_id": "context-1",
            "source_assets": [{"asset_id": "asset-1", "sha256": "sha-1"}],
            "document_ids": ["document-1"],
            "evidence_refs": [evidence_uri],
        }).uri
        config = root / "dataforge.toml"
        config.write_text(
            "context_limit_tokens=512\n"
            "[models.generator]\nmodel='fixture'\nbackend='fixture'\nprofile='test'\nmax_output_tokens=64\n"
            "[splitting]\nseed='seed'\n"
            "[qualification]\nmax_attempts=2\n",
            encoding="utf-8",
        )
        raw = build_dataset(source_uri, "handoff", "paragraph-qa", config, runtime=runtime, inference_client=FixtureInferenceClient())
        qualified = build_dataset(source_uri, "handoff", "paragraph-qa-qualified", config, runtime=runtime, inference_client=FixtureInferenceClient())
        raw_manifest = _read_json(runtime, raw["dataset_manifest_uri"])
        qualified_manifest = _read_json(runtime, qualified["dataset_manifest_uri"])
        raw_candidate_uri = raw_manifest["records_uri"].replace("records.jsonl", "candidates.jsonl")
        raw_candidate = _read_jsonl(runtime, raw_candidate_uri)[0]
        qualified_candidate = _read_jsonl(runtime, qualified_manifest["qualification_artifacts"]["candidates"]["uri"])[0]
        for field in ("candidate_id", "source_text", "question", "reference"):
            assert raw_candidate[field] == qualified_candidate[field]

        exact = create_exact_recall_set(runtime, qualified["dataset_manifest_uri"])
        paraphrase = import_evaluation_set(
            runtime,
            REPOSITORY / "design_input/ift_research_foundation_v1/empirical/evaluation_sets/paraphrase_import_fixture_v1.jsonl",
            evaluation_set_name="paraphrase-import-v1",
            research_role="paraphrase_evaluation",
        )
        package = create_research_package(
            runtime,
            package_name="cpu-handoff",
            dataset_manifest_uri=qualified["dataset_manifest_uri"],
            evaluation_manifest_uris=[exact["manifest_uri"], paraphrase["manifest_uri"]],
        )

        reader = DataForgeDatasetReader(package["manifest_uri"], storage_runtime=runtime, input_mode="dataforge_manifest")
        training_records = list(reader.iter_training_records())
        evaluation_records = list(reader.iter_evaluation_records())
        assert training_records and all(record.metadata.get("research_role") == "training" for record in training_records)
        assert {record.metadata.get("research_role") for record in evaluation_records} == {"exact_recall", "paraphrase_evaluation"}

        ids = build_lineage_ids(
            {"schema_version": "cpu-handoff/v1", "training": {"max_steps": 0}, "research_package": package["research_package_version"]},
            requested_experiment_id="exp-cpu-handoff",
            requested_run_id="cpu-handoff",
        )
        base_model = {"name": "fixture-base", "resolved_revision": "fixture"}
        outputs = [{
            "record_id": record.record_id,
            "prompt": record.messages[0]["content"],
            "expected": record.messages[-1]["content"],
            "generated": "fixture prediction",
            "exact_match": False,
            "contains_expected": False,
            "metadata": record.metadata,
            "provenance": {},
        } for record in evaluation_records]
        baseline_predictions = prediction_rows({"outputs": outputs}, prediction_type="baseline", ids=ids, base_model_identity=base_model, decoding={"do_sample": False})
        trained_predictions = prediction_rows({"outputs": outputs}, prediction_type="trained", ids=ids, base_model_identity=base_model, decoding={"do_sample": False})
        assert {row["research_role"] for row in baseline_predictions} == {"exact_recall", "paraphrase_evaluation"}

        staging = root / "adapter"
        staging.mkdir()
        staging.joinpath("adapter_config.json").write_text('{"peft_type":"LORA"}\n', encoding="utf-8")
        staging.joinpath("adapter_model.safetensors").write_bytes(b"fixture-adapter")
        publisher = TrainingPublisher(runtime, ids, experiment_name="CPU research handoff")
        lineage = reader.lineage()
        variant_identity = {"schema_version": "cpu-handoff/v1", "training": {"max_steps": 0}}
        publisher.publish_experiment()
        publisher.publish_variant(variant_identity, dataset_lineage=lineage, base_model_identity=base_model)
        publisher.publish_training_request(dataset_lineage=lineage, normalized_request=variant_identity["training"], base_model_identity=base_model, publication_mode="storage")
        publication = publisher.publish_completed_run(
            staging_directory=staging,
            dataset_lineage=lineage,
            base_model_identity=base_model,
            adapter_details={"type": "fixture", "format": "peft"},
            resolved_config=variant_identity,
            environment={"execution": "cpu-fixture"},
            training_report={"status": "completed", **ids.to_dict()},
            metrics={"train_steps": 0, "evaluation_examples": len(evaluation_records)},
            baseline_predictions=baseline_predictions,
            trained_predictions=trained_predictions,
            retain_local_staging=False,
        )
        tracking_payload = completed_run_payload(
            identity=ids.to_dict(),
            parameters=variant_identity,
            metrics={"evaluation_examples": len(evaluation_records)},
            resources={},
            artifact_references={"publication_manifest_uri": publication.publication_manifest_uri, "adapter_uri": publication.adapter_uri},
            artifact_checksums=publication.artifact_checksums,
        )
        assert tracking_payload["artifact_references"]["publication_manifest_uri"].startswith("storage://")
        assert NoOpTracker().log_completed_run(tracking_payload).status == "disabled"
        print(json.dumps({
            "qualified_training_records": len(training_records),
            "evaluation_records": len(evaluation_records),
            "evaluation_roles": sorted({row["research_role"] for row in baseline_predictions}),
            "publication_manifest_uri": publication.publication_manifest_uri,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
