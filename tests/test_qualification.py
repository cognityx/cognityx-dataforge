from __future__ import annotations

import json
from pathlib import Path

import pytest
from cognityx_jobs import JobRepository
from cognityx_storage import StorageConfig, StorageRuntime

from cognityx_dataforge.build import build_dataset
from cognityx_dataforge.qualification import (
    QualificationPipeline,
    qualification_decision,
)
from cognityx_dataforge.source import resolve_storage_uri


PACK = Path(__file__).parents[1] / "design_input" / "ift_research_foundation_v1"
FIXTURES = Path(__file__).parent / "fixtures" / "qualification"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stage_input(request: dict) -> dict:
    prompt = request["messages"][-1]["content"]
    return json.loads(prompt.split("\n\nINPUT:\n", 1)[1])


def test_frozen_historical_oracle_decisions():
    cases = {row["reference_id"]: row for row in _rows(PACK / "empirical/qualification/e3_qa_qualification_cases_v1.jsonl")}
    expected = _rows(PACK / "empirical/qualification/expected_qualification_v1.jsonl")
    for oracle in expected:
        case = cases[oracle["reference_id"]]
        decision = qualification_decision(
            oracle["expected_question_demand"],
            oracle["expected_source_answerability"],
            oracle["expected_reference_qualification"],
            case["gold_reference"],
        )
        assert decision["decision"] == oracle["expected_decision"]
        assert decision["reason_codes"] == oracle["expected_reason_codes"]
        assert decision["rewrite_allowed"] is False


def test_deterministic_mutation_reason_codes():
    mutations = {row["mutation_id"]: row for row in _rows(PACK / "empirical/qualification/deterministic_mutations_v1.jsonl")}
    numeric_requirements = {"required_slots": ["sustained_wind_stop", "gust_stop", "logical_relation"], "answer_structure": "numeric_role_rule"}
    numeric_source = {
        "answerable_at_requested_specificity": True,
        "slot_values": {
            "sustained_wind_stop": {"value": 23, "unit": "knots"},
            "gust_stop": {"value": 31, "unit": "knots", "count": 1},
            "logical_relation": "OR",
        },
        "source_text": mutations["M001"]["source"],
    }
    complete = {"answers_question": True, "required_slot_coverage": 1.0, "unsupported_claims": [], "contradicted_claims": []}
    swapped = qualification_decision(numeric_requirements, numeric_source, complete, mutations["M001"]["reference"])
    assert swapped["reason_codes"] == ["numeric_role_binding_failed"]

    malformed = qualification_decision(None, None, None, mutations["M006"]["reference"], infrastructure_uncertainty=True)
    assert malformed["decision"] == "needs_review"
    assert malformed["reason_codes"] == mutations["M006"]["expected_reason_codes"]

    unresolved = qualification_decision(
        {"required_slots": ["documented_reason"], "answer_structure": "short_factual_rule"},
        {"answerable_at_requested_specificity": True, "slot_values": {"documented_reason": "actual documented reason"}},
        complete,
        mutations["M005"]["reference"],
        provenance_resolvable=False,
    )
    assert unresolved["reason_codes"] == mutations["M005"]["expected_reason_codes"]

    unsupported = qualification_decision(
        numeric_requirements,
        {**numeric_source, "source_text": mutations["M009"]["source"]},
        {
            **complete,
            "source_faithfulness": "failed",
            "unsupported_claims": ["the threshold also applies to higher values"],
        },
        mutations["M009"]["reference"],
    )
    assert "unsupported_claim" in unsupported["reason_codes"]

    missing_qualifier = qualification_decision(
        {"required_slots": ["subject_location"], "answer_structure": "constraint_rule"},
        {"answerable_at_requested_specificity": True, "slot_values": {"subject_location": "subject line"}, "source_text": mutations["M002"]["source"]},
        {**complete, "mandatory_qualifiers_present": False},
        mutations["M002"]["reference"],
    )
    assert missing_qualifier["reason_codes"] == mutations["M002"]["expected_reason_codes"]

    partial_phrase = qualification_decision(
        {"required_slots": ["integrity_phrase"], "answer_structure": "exact_phrase"},
        {"answerable_at_requested_specificity": True, "slot_values": {"integrity_phrase": "glass mango over silent river"}, "source_text": mutations["M003"]["source"]},
        {**complete, "exact_phrase_complete": False},
        mutations["M003"]["reference"],
    )
    assert partial_phrase["reason_codes"] == mutations["M003"]["expected_reason_codes"]

    normalized_number = qualification_decision(
        {"required_slots": ["paid_hours_minimum"], "answer_structure": "numeric_rule"},
        {"answerable_at_requested_specificity": True, "slot_values": {"paid_hours_minimum": {"value": 4, "unit": "hours"}}, "source_text": mutations["M004"]["source"]},
        complete,
        mutations["M004"]["reference"],
    )
    assert normalized_number["decision"] == mutations["M004"]["expected"]

    list_failure = qualification_decision(
        {"required_slots": ["policy_scope_members"], "answer_structure": "complete_set"},
        {"answerable_at_requested_specificity": True, "slot_values": {"policy_scope_members": ["employees", "apprentices", "contractors", "visitors with brass tokens", "registered companion creatures"]}, "source_text": mutations["M007"]["source"]},
        {**complete, "missing_required_members": ["apprentices", "registered companion creatures"], "unsupported_members": ["temporary workers"]},
        mutations["M007"]["reference"],
    )
    assert list_failure["reason_codes"] == mutations["M007"]["expected_reason_codes"]

    unanswerable = qualification_decision(
        {"required_slots": ["approval_type_or_authority"], "answer_structure": "identity_or_type"},
        {"answerable_at_requested_specificity": False, "slot_values": {}, "missing_slots": ["approval_type_or_authority"], "source_text": mutations["M010"]["source"]},
        {"answers_question": True, "required_slot_coverage": 0.0, "unsupported_claims": [], "unsupported_reference_claims": ["immediate supervisor"], "contradicted_claims": []},
        mutations["M010"]["reference"],
    )
    assert unanswerable["reason_codes"] == mutations["M010"]["expected_reason_codes"]


def test_generic_finance_policy_uses_only_declarative_requirements():
    fixture = json.loads(
        (FIXTURES / "finance_policy_generic_v1.json").read_text(encoding="utf-8")
    )
    accepted = qualification_decision(
        fixture["answer_requirements"],
        fixture["source_answerability"],
        fixture["reference_qualification"],
        fixture["reference"],
    )
    assert accepted["decision"] == "accepted"
    assert accepted["reference_correctness"] == "correct"
    assert accepted["source_faithfulness"] == "passed"

    rejected = qualification_decision(
        fixture["answer_requirements"],
        fixture["source_answerability"],
        {
            **fixture["reference_qualification"],
            "required_slot_coverage": 0.5,
            "reference_correctness": "partially_correct",
            "reference_completeness": "incomplete",
            "supported_claims": ["one approval is required"],
            "unsupported_claims": ["the budget analyst must approve"],
            "missing_required_members": ["regional finance controller"],
            "unsupported_members": ["budget analyst"],
        },
        "It requires one approval from the cost-centre owner and a budget analyst.",
    )
    assert rejected["decision"] == "rejected"
    assert rejected["reference_correctness"] == "partially_correct"
    assert rejected["reference_completeness"] == "incomplete"
    assert rejected["source_faithfulness"] == "failed"
    assert set(rejected["reason_codes"]) >= {
        "missing_required_facts",
        "numeric_role_binding_failed",
        "missing_required_members",
        "unsupported_members",
        "unsupported_claim",
    }


def test_production_qualification_has_no_pilot_role_or_phrase_rules():
    source = Path(QualificationPipeline.__module__.replace(".", "/"))
    source = Path(__file__).parents[1] / "src" / source.with_suffix(".py")
    text = source.read_text(encoding="utf-8").lower()
    for forbidden in (
        "sustained_wind_stop",
        "gust_stop",
        "paid_hours_minimum",
        '"sustained"',
        '"gust"',
        '"paid hours"',
        '"or higher"',
    ):
        assert forbidden not in text


def test_quality_axes_do_not_collapse_operational_rejection():
    decision = qualification_decision(
        {
            "question_validity": "valid",
            "required_slots": ["policy_limit"],
            "answer_structure": "numeric_rule",
            "allowed_inference_policy": "source_explicit_only",
            "requirements": [{
                "requirement_id": "policy_limit",
                "semantic_role": "policy limit",
                "semantic_role_terms": ["limit"],
                "value_type": "number",
                "unit": "days",
            }],
        },
        {
            "answerable_at_requested_specificity": True,
            "requirement_bindings": [{
                "requirement_id": "policy_limit",
                "expected_value": 5,
                "value_type": "number",
                "unit": "days",
            }],
        },
        {
            "answers_question": True,
            "required_slot_coverage": 1.0,
            "reference_correctness": "correct",
            "reference_completeness": "complete",
            "source_faithfulness": "failed",
            "supported_claims": ["the limit is five days"],
            "unsupported_claims": ["the limit automatically renews"],
            "contradicted_claims": [],
        },
        "The limit is five days and it automatically renews.",
    )
    assert decision["decision"] == "rejected"
    assert decision["quality_label"] == "correct"
    assert decision["reference_correctness"] == "correct"
    assert decision["source_faithfulness"] == "failed"
    assert decision["operational_acceptance"] == "rejected"

    explicit_failure = qualification_decision(
        {
            "question_validity": "valid",
            "required_slots": [],
            "answer_structure": "short_factual_rule",
        },
        {"answerable_at_requested_specificity": True},
        {
            "answers_question": True,
            "required_slot_coverage": 1.0,
            "reference_correctness": "correct",
            "reference_completeness": "complete",
            "source_faithfulness": "failed",
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
        },
        "A factually correct answer with a separately failed support check.",
    )
    assert explicit_failure["decision"] == "rejected"
    assert explicit_failure["reason_codes"] == ["source_faithfulness_failed"]
    assert explicit_failure["reference_correctness"] == "correct"


def test_uncertain_question_validity_is_reviewed_not_accepted():
    decision = qualification_decision(
        {
            "question_validity": "uncertain",
            "required_slots": [],
            "answer_structure": "short_factual_rule",
        },
        {"answerable_at_requested_specificity": True},
        {
            "answers_question": True,
            "required_slot_coverage": 1.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
        },
        "An otherwise acceptable answer.",
    )
    assert decision["question_validity"] == "uncertain"
    assert decision["reference_correctness"] == "correct"
    assert decision["operational_acceptance"] == "needs_review"


class QualificationClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if "Analyze only the question" in prompt:
            payload = {"required_slots": ["documented_reason"], "answer_structure": "short_factual_rule"}
        elif "Use the frozen question requirements" in prompt:
            payload = {
                "answerable_at_requested_specificity": True,
                "slot_values": {"documented_reason": "actual documented reason for a decision"},
                "missing_slots": [],
            }
        elif "Compare the generated reference" in prompt:
            payload = {
                "answers_question": True,
                "required_slot_coverage": 1.0,
                "unsupported_claims": [],
                "contradicted_claims": [],
                "premise_restatement": False,
            }
        else:
            payload = {
                "instruction": "Under the Clear Moon Principle, what must a manager state?",
                "answer": "The actual documented reason for a decision.",
            }
        return {"id": f"request-{len(self.requests)}", "choices": [{"message": {"content": json.dumps(payload)}}]}

    def count_input_tokens(self, **kwargs):
        return 20


class UncertainQualificationClient(QualificationClient):
    def chat(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Analyze only the question" in prompt:
            self.requests.append(kwargs)
            return {"choices": [{"message": {"content": "{not valid json"}}]}
        return super().chat(**kwargs)


class UnanswerableApprovalClient(QualificationClient):
    def chat(self, **kwargs):
        self.requests.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        if "Analyze only the question" in prompt:
            payload = {"required_slots": ["approval_type_or_authority"], "answer_structure": "identity_or_type"}
        elif "Use the frozen question requirements" in prompt:
            payload = {
                "answerable_at_requested_specificity": False,
                "slot_values": {},
                "missing_slots": ["approval_type_or_authority"],
            }
        elif "Compare the generated reference" in prompt:
            payload = {
                "answers_question": False,
                "required_slot_coverage": 0.0,
                "unsupported_claims": [],
                "contradicted_claims": [],
                "premise_restatement": True,
            }
        else:
            payload = {
                "instruction": "Before exceeding 11.5 overtime hours, what approval is required?",
                "answer": "Approval is required before exceeding 11.5 overtime hours.",
            }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _qualified_fixture(tmp_path: Path, *, source_text: str | None = None):
    runtime = StorageRuntime.from_config(StorageConfig.built_in(root=tmp_path / "storage"))
    artifact = runtime.for_role("artifact")
    evidence = {
        "evidence_id": "evidence-clear-moon",
        "document_id": "document-1",
        "source_asset_id": "asset-1",
        "page_number": 3,
        "physical_page_index": 2,
        "printed_page_label": "1",
        "pdf_page_label": "3",
        "text": source_text or "PR-01 Clear Moon Principle. A manager must state the actual documented reason for a decision.",
        "char_start": 0,
        "char_end": 94,
        "anchor_id": "document-1:page-index:2",
        "block_id": "document-1:page-index:2:block:1",
        "source_sha256": "sha-1",
        "context_id": "context-1",
        "run_id": "ingest-run-qualified",
    }
    evidence_uri = artifact.put_bytes(
        "ingest/qualified/evidence.jsonl",
        (json.dumps(evidence) + "\n").encode(),
        media_type="application/x-ndjson",
    ).uri
    manifest_uri = artifact.put_json("ingest/qualified/manifest.json", {
        "schema": "cognityx.ingest.run",
        "run_id": "ingest-run-qualified",
        "context_id": "context-1",
        "source_assets": [{"asset_id": "asset-1", "sha256": "sha-1"}],
        "document_ids": ["document-1"],
        "evidence_refs": [evidence_uri],
    }).uri
    config = tmp_path / "dataforge.toml"
    config.write_text(
        "context_limit_tokens=512\n"
        "[models.generator]\nmodel='fake'\nbackend='fake'\nprofile='test'\nmax_output_tokens=64\n"
        "[splitting]\nseed='seed'\n"
        "[qualification]\nmax_attempts=2\n",
        encoding="utf-8",
    )
    return runtime, manifest_uri, config


def test_qualified_recipe_is_candidate_blind_and_persists_all_stages(tmp_path: Path):
    runtime, manifest_uri, config = _qualified_fixture(tmp_path)
    raw_result = build_dataset(
        manifest_uri,
        "qualified-demo",
        "paragraph-qa",
        config,
        runtime=runtime,
        jobs=JobRepository(":memory:"),
        inference_client=QualificationClient(),
    )
    client = QualificationClient()
    result = build_dataset(
        manifest_uri,
        "qualified-demo",
        "paragraph-qa-qualified",
        config,
        runtime=runtime,
        jobs=JobRepository(":memory:"),
        inference_client=client,
    )
    assert result["record_count"] == 1
    manifest_store, manifest_key = resolve_storage_uri(runtime, result["dataset_manifest_uri"], role_name="dataset")
    with manifest_store.open(manifest_key) as handle:
        manifest = json.load(handle)
    assert manifest["accepted_count"] == 1
    assert manifest["needs_review_count"] == 0
    assert set(manifest["qualification_artifacts"]) >= {
        "candidates", "answer-requirements", "source-answerability",
        "reference-qualification", "qualification-decisions", "accepted",
        "rejected", "needs-review", "model-calls",
    }
    records_store, records_key = resolve_storage_uri(runtime, manifest["records_uri"], role_name="dataset")
    with records_store.open(records_key) as handle:
        record = json.loads(handle.readline())
    assert record["metadata"]["research_role"] == "training"
    assert record["metadata"]["training_eligible"] is True
    coordinates = record["metadata"]["page_coordinates"]
    assert coordinates["physical_page_index"] == 2
    assert coordinates["printed_page_label"] == "1"
    assert coordinates["physical_page_index"] != coordinates["printed_page_label"]

    raw_manifest_store, raw_manifest_key = resolve_storage_uri(runtime, raw_result["dataset_manifest_uri"], role_name="dataset")
    with raw_manifest_store.open(raw_manifest_key.removesuffix("manifest.json") + "candidates.jsonl") as handle:
        raw_candidate = json.loads(handle.readline())
    qualified_candidate_uri = manifest["qualification_artifacts"]["candidates"]["uri"]
    qualified_store, qualified_key = resolve_storage_uri(runtime, qualified_candidate_uri, role_name="dataset")
    with qualified_store.open(qualified_key) as handle:
        qualified_candidate = json.loads(handle.readline())
    shared_fields = ("candidate_id", "evidence_id", "char_start", "char_end", "source_text", "question", "reference")
    assert {key: raw_candidate[key] for key in shared_fields} == {
        key: qualified_candidate[key] for key in shared_fields
    }

    by_role = {request["request_metadata"]["model_role"]: request for request in client.requests}
    stage_a = by_role["answer_requirements"]["messages"][-1]["content"]
    stage_b = by_role["source_answerability"]["messages"][-1]["content"]
    stage_c = by_role["reference_qualification"]["messages"][-1]["content"]
    assert "generated_reference" not in stage_a
    assert "actual documented reason for a decision" not in stage_a.lower()
    assert "generated_reference" not in stage_b
    assert "The actual documented reason for a decision." not in stage_b
    assert set(_stage_input(by_role["answer_requirements"])) == {"question"}
    assert set(_stage_input(by_role["source_answerability"])) == {
        "answer_requirements", "provenance", "source_evidence",
    }
    stage_c_input = _stage_input(by_role["reference_qualification"])
    assert set(stage_c_input) == {
        "answer_requirements", "source_answerability", "generated_reference",
    }
    assert "source_text" not in json.dumps(stage_c_input)
    assert "PR-01 Clear Moon Principle" not in stage_c

    source_artifact_uri = manifest["qualification_artifacts"]["source-answerability"]["uri"]
    source_store, source_key = resolve_storage_uri(runtime, source_artifact_uri, role_name="dataset")
    with source_store.open(source_key) as handle:
        frozen_source_result = json.loads(handle.readline())
    assert "source_text" not in json.dumps(frozen_source_result)

    reused = build_dataset(
        manifest_uri,
        "qualified-demo",
        "paragraph-qa-qualified",
        config,
        runtime=runtime,
        jobs=JobRepository(":memory:"),
        inference_client=client,
    )
    assert reused.get("reused") is True, (result, reused)


def test_review_and_rejection_are_durable_and_never_rewritten(tmp_path: Path):
    runtime, manifest_uri, config = _qualified_fixture(tmp_path)
    review = build_dataset(
        manifest_uri,
        "review-demo",
        "paragraph-qa-qualified",
        config,
        runtime=runtime,
        inference_client=UncertainQualificationClient(),
    )
    review_store, review_manifest_key = resolve_storage_uri(runtime, review["dataset_manifest_uri"], role_name="dataset")
    with review_store.open(review_manifest_key) as handle:
        review_manifest = json.load(handle)
    assert review_manifest["accepted_count"] == 0
    assert review_manifest["needs_review_count"] == 1
    review_uri = review_manifest["qualification_artifacts"]["needs-review"]["uri"]
    review_rows_store, review_rows_key = resolve_storage_uri(runtime, review_uri, role_name="dataset")
    with review_rows_store.open(review_rows_key) as handle:
        review_row = json.loads(handle.readline())
    assert review_row["reason_codes"] == ["qualification_infrastructure_uncertainty"]

    rejected = build_dataset(
        manifest_uri,
        "rejected-demo",
        "paragraph-qa-qualified",
        config,
        runtime=runtime,
        inference_client=UnanswerableApprovalClient(),
    )
    rejected_store, rejected_manifest_key = resolve_storage_uri(runtime, rejected["dataset_manifest_uri"], role_name="dataset")
    with rejected_store.open(rejected_manifest_key) as handle:
        rejected_manifest = json.load(handle)
    rejected_uri = rejected_manifest["qualification_artifacts"]["rejected"]["uri"]
    rejected_rows_store, rejected_rows_key = resolve_storage_uri(runtime, rejected_uri, role_name="dataset")
    with rejected_rows_store.open(rejected_rows_key) as handle:
        rejected_row = json.loads(handle.readline())
    assert rejected_row["reason_codes"] == [
        "source_not_answerable_at_requested_specificity",
        "reference_does_not_answer_question",
        "premise_restatement",
    ]
    serialized = json.dumps(rejected_row).lower()
    assert "supervisor" not in serialized
    assert "hr manager" not in serialized


class InterruptSecondQualificationClient(QualificationClient):
    def __init__(self) -> None:
        super().__init__()
        self.requirement_calls = 0

    def chat(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Analyze only the question" in prompt:
            self.requirement_calls += 1
            if self.requirement_calls == 2:
                raise SystemExit("fixture interruption")
        return super().chat(**kwargs)


def test_qualified_recipe_resumes_from_per_record_results(tmp_path: Path):
    runtime, manifest_uri, config = _qualified_fixture(
        tmp_path,
        source_text=(
            "PR-01 Clear Moon Principle. A manager must state the actual documented reason for a decision."
            "\n\nThe policy also applies to employees and apprentices."
        ),
    )
    interrupted = InterruptSecondQualificationClient()
    with pytest.raises(SystemExit, match="fixture interruption"):
        build_dataset(
            manifest_uri,
            "resume-demo",
            "paragraph-qa-qualified",
            config,
            runtime=runtime,
            jobs=JobRepository(":memory:"),
            inference_client=interrupted,
        )
    assert interrupted.requirement_calls == 2

    resumed_client = QualificationClient()
    result = build_dataset(
        manifest_uri,
        "resume-demo",
        "paragraph-qa-qualified",
        config,
        runtime=runtime,
        jobs=JobRepository(":memory:"),
        inference_client=resumed_client,
    )
    requirement_calls = [
        request for request in resumed_client.requests
        if request["request_metadata"]["model_role"] == "answer_requirements"
    ]
    assert len(requirement_calls) == 1
    assert result["record_count"] == 1
