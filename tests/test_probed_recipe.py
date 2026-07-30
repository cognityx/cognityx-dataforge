import json
from pathlib import Path

from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset


class ProbedClient:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if "knowledge_units" in system:
            payload = {"knowledge_units": [{"canonical_statement": "Water freezes at zero degrees.", "supporting_facts": ["Water freezes at zero degrees."], "concepts": ["water"]}]}
        elif "own knowledge" in system:
            payload = {"answer": "I do not know."}
        elif "question" in system:
            payload = {"question": "At what temperature does water freeze?"}
        elif "class (" in system:
            payload = {"class": "unknown", "reasons": {"missing_fact": True}, "reference_answer": "Water freezes at zero degrees."}
        elif "decision accept" in system:
            payload = {"decision": "accept", "reasons": {"grounded": True}}
        else:
            payload = {"instruction": "At what temperature does water freeze?", "answer": "Water freezes at zero degrees."}
        return {"id": f"request-{len(self.calls)}", "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}], "cognityx": {"token_budget": {"effective_max_output_tokens": 32}}}


def test_probed_recipe_selects_unknown_and_keeps_student_prompt_evidence_free(tmp_path: Path):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    store = runtime.for_role("artifact")
    evidence = {"evidence_id": "e1", "document_id": "doc-1", "source_asset_id": "asset-1", "page_number": 1, "text": "Water freezes at zero degrees.", "char_start": 0, "char_end": 30, "context_id": "ctx-1", "run_id": "run-1"}
    evidence_uri = store.put_bytes("ingest/documents/doc-1/evidence.jsonl", (json.dumps(evidence) + "\n").encode(), media_type="application/x-ndjson").uri
    manifest_uri = store.put_json("ingest/runs/run-1/manifest.json", {"schema": "cognityx.ingest.run", "run_id": "run-1", "context_id": "ctx-1", "document_ids": ["doc-1"], "evidence_refs": [evidence_uri]}).uri
    config = tmp_path / "config.toml"
    config.write_text("""[models.generator]\nmodel='m'\nmax_output_tokens=32\n[probing]\nprobes_per_unit=2\ninclude_classes=['unknown']\n""", encoding="utf-8")
    client = ProbedClient()
    result = build_dataset(manifest_uri, "probed", "knowledge-unit-probed-qa", config, runtime=runtime, inference_client=client)
    dataset = runtime.for_role("dataset")
    version = result["dataset_manifest_uri"].rsplit("/", 2)[-2]
    root = f"{result['dataset_id']}/{version}"
    with dataset.open(f"{root}/records.jsonl") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len(records) == 1
    assert all(record["metadata"]["probe_class"] == "unknown" for record in records)
    with dataset.open(f"{root}/manifest.json") as handle:
        assert json.load(handle)["duplicate_count"] == 1
    student_calls = [call for call in client.calls if "own knowledge" in call["messages"][0]["content"]]
    assert len(student_calls) == 2
    assert all("Water freezes" not in call["messages"][1]["content"] for call in student_calls)
    validation_calls = [call for call in client.calls if "decision accept" in call["messages"][0]["content"]]
    assert validation_calls
    validation_prompt = validation_calls[0]["messages"][1]["content"]
    assert all(value in validation_prompt for value in ("ORIGINAL EVIDENCE", "KNOWLEDGE UNIT", "PROBE QUESTION", "STUDENT RESPONSE", "PROBE JUDGMENT", "CANDIDATE"))
    assert validation_calls[0]["execution_context"]["recipe"] == "knowledge-unit-probed-qa"
    assert validation_calls[0]["request_metadata"]["history_mode"] == "none"
