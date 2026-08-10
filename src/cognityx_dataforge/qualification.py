from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cognityx_dataforge.inference import GeneratorConfig, StructuredAdapter


ANSWER_REQUIREMENTS_SCHEMA = "cognityx.dataforge.answer-requirements/v1"
SOURCE_ANSWERABILITY_SCHEMA = "cognityx.dataforge.source-answerability/v1"
REFERENCE_QUALIFICATION_SCHEMA = "cognityx.dataforge.reference-qualification/v1"
QUALIFICATION_DECISION_SCHEMA = "cognityx.dataforge.qualification-decision/v1"

DECISIONS = frozenset({"accepted", "rejected", "needs_review"})


class QualificationOutputError(ValueError):
    """Raised when a qualification stage returns unusable structured output."""


def _json_object(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise QualificationOutputError("qualification output must be a JSON object")
    return value


def _requirements(raw: str) -> dict[str, Any]:
    value = _json_object(raw)
    if not isinstance(value.get("required_slots"), list) or not value.get("answer_structure"):
        raise QualificationOutputError("answer requirements need required_slots and answer_structure")
    specifications = value.get("requirements")
    if specifications is None:
        specifications = [
            {
                "requirement_id": str(slot),
                "semantic_role": str(slot),
                "value_type": "unspecified",
            }
            for slot in value["required_slots"]
        ]
    if not isinstance(specifications, list) or any(
        not isinstance(item, dict) or not item.get("requirement_id")
        for item in specifications
    ):
        raise QualificationOutputError("answer requirements requirements must be a list of identified objects")
    identifiers = [str(item["requirement_id"]) for item in specifications]
    if len(identifiers) != len(set(identifiers)):
        raise QualificationOutputError("answer requirement IDs must be unique")
    question_validity = value.get("question_validity", "valid")
    if question_validity not in {"valid", "invalid", "uncertain"}:
        raise QualificationOutputError("question_validity must be valid, invalid, or uncertain")
    return {
        "schema": ANSWER_REQUIREMENTS_SCHEMA,
        **value,
        "question_validity": question_validity,
        "allowed_inference_policy": value.get(
            "allowed_inference_policy", "source_explicit_only"
        ),
        "requirements": specifications,
    }


def _contains_forbidden_source_field(value: Any) -> bool:
    forbidden = {"source_text", "source_evidence", "raw_source_text", "evidence_text"}
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_forbidden_source_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_source_field(item) for item in value)
    return False


def _answerability(raw: str) -> dict[str, Any]:
    value = _json_object(raw)
    if not isinstance(value.get("answerable_at_requested_specificity"), bool):
        raise QualificationOutputError("source answerability needs a boolean answerable_at_requested_specificity")
    if not isinstance(value.get("slot_values", {}), dict):
        raise QualificationOutputError("source answerability slot_values must be an object")
    if _contains_forbidden_source_field(value):
        raise QualificationOutputError(
            "source answerability must freeze supported facts and constraints, not raw source text"
        )
    bindings = value.get("requirement_bindings")
    if bindings is None:
        bindings = []
        for requirement_id, expected in value.get("slot_values", {}).items():
            binding: dict[str, Any] = {
                "requirement_id": str(requirement_id),
                "supported": True,
            }
            if isinstance(expected, dict) and "value" in expected:
                binding.update(
                    expected_value=expected["value"],
                    value_type=expected.get("value_type", "number"),
                    unit=expected.get("unit"),
                    relation=expected.get("relation"),
                    cardinality=expected.get("cardinality", expected.get("count")),
                )
            elif isinstance(expected, list):
                binding.update(value_type="set", expected_members=expected)
            else:
                binding.update(expected_value=expected, value_type="text")
            bindings.append(binding)
    if not isinstance(bindings, list) or any(
        not isinstance(item, dict) or not item.get("requirement_id")
        for item in bindings
    ):
        raise QualificationOutputError(
            "source answerability requirement_bindings must be a list of identified objects"
        )
    return {
        "schema": SOURCE_ANSWERABILITY_SCHEMA,
        **value,
        "requirement_bindings": bindings,
        "supported_claims": list(value.get("supported_claims", [])),
        "evidence_anchors": list(value.get("evidence_anchors", [])),
    }


def _reference_qualification(raw: str) -> dict[str, Any]:
    value = _json_object(raw)
    if not isinstance(value.get("answers_question"), bool):
        raise QualificationOutputError("reference qualification needs a boolean answers_question")
    coverage = value.get("required_slot_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise QualificationOutputError("reference qualification needs numeric required_slot_coverage")
    if not 0.0 <= float(coverage) <= 1.0:
        raise QualificationOutputError("required_slot_coverage must be between 0 and 1")
    for name in (
        "supported_claims",
        "unsupported_claims",
        "contradicted_claims",
        "missing_required_facts",
    ):
        if not isinstance(value.get(name, []), list):
            raise QualificationOutputError(f"reference qualification {name} must be a list")
    return {
        "schema": REFERENCE_QUALIFICATION_SCHEMA,
        **value,
        "supported_claims": list(value.get("supported_claims", [])),
        "unsupported_claims": list(value.get("unsupported_claims", [])),
        "contradicted_claims": list(value.get("contradicted_claims", [])),
        "missing_required_facts": list(value.get("missing_required_facts", [])),
        "requirement_results": list(value.get("requirement_results", [])),
    }


def _words_to_digits(value: str) -> str:
    words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12",
    }
    return re.sub(
        r"\b(" + "|".join(words) + r")\b",
        lambda match: words[match.group(1).lower()],
        value.lower(),
        flags=re.IGNORECASE,
    )


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", _words_to_digits(str(value))).strip()


_GENERIC_ROLE_WORDS = frozenset({
    "amount", "expected", "limit", "maximum", "minimum", "number", "numeric",
    "required", "requirement", "rule", "stop", "threshold", "value",
})


def _semantic_role_terms(specification: dict[str, Any]) -> list[str]:
    declared = specification.get("semantic_role_terms") or specification.get("role_terms")
    if declared:
        return [_normalized(item) for item in declared if _normalized(item)]
    role = specification.get("semantic_role") or specification.get("requirement_id", "")
    return [
        token
        for token in _normalized(role).split()
        if token not in _GENERIC_ROLE_WORDS
    ]


def _numeric_role_present(
    reference: str,
    *,
    value: Any,
    specification: dict[str, Any],
) -> bool:
    tokens = [token.strip(".") for token in _normalized(reference).split()]
    number = _normalized(value)
    number_positions = [index for index, token in enumerate(tokens) if token == number]
    if not number_positions:
        return False
    role_terms = _semantic_role_terms(specification)
    unit_terms = _normalized(specification.get("unit", "")).split()
    for position in number_positions:
        window = tokens[max(0, position - 3):position + 4]
        role_matches = not role_terms or any(term in window for term in role_terms)
        unit_matches = not unit_terms or all(
            any(token.rstrip("s") == term.rstrip("s") for token in window)
            for term in unit_terms
        )
        if role_matches and unit_matches:
            return True
    return False


def _requirement_specifications(
    requirements: dict[str, Any],
    answerability: dict[str, Any],
) -> list[dict[str, Any]]:
    declared = requirements.get("requirements")
    if not isinstance(declared, list):
        declared = [
            {
                "requirement_id": str(slot),
                "semantic_role": str(slot),
                "value_type": "unspecified",
            }
            for slot in requirements.get("required_slots", [])
        ]
    bindings = {
        str(item.get("requirement_id")): item
        for item in answerability.get("requirement_bindings", [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    slot_values = answerability.get("slot_values", {})
    specifications: list[dict[str, Any]] = []
    for item in declared:
        specification = dict(item)
        requirement_id = str(specification.get("requirement_id"))
        specification.update(bindings.get(requirement_id, {}))
        if requirement_id in slot_values and "expected_value" not in specification:
            expected = slot_values[requirement_id]
            if isinstance(expected, dict) and "value" in expected:
                specification.update(
                    expected_value=expected["value"],
                    value_type=expected.get("value_type", "number"),
                    unit=expected.get("unit", specification.get("unit")),
                    relation=expected.get("relation", specification.get("relation")),
                    cardinality=expected.get(
                        "cardinality", expected.get("count", specification.get("cardinality"))
                    ),
                )
            elif isinstance(expected, list):
                specification.update(value_type="set", expected_members=expected)
            else:
                specification["expected_value"] = expected
        specifications.append(specification)
    return specifications


def deterministic_reference_checks(
    requirements: dict[str, Any],
    answerability: dict[str, Any],
    qualification: dict[str, Any],
    reference: str,
    *,
    provenance_resolvable: bool = True,
) -> list[str]:
    """Return stable failure codes that do not depend on another model call."""

    reasons: list[str] = []
    if not provenance_resolvable:
        reasons.append("provenance_unresolvable")

    normalized_reference = _normalized(reference)
    qualification_results = {
        str(item.get("requirement_id")): item
        for item in qualification.get("requirement_results", [])
        if isinstance(item, dict) and item.get("requirement_id")
    }
    for specification in _requirement_specifications(requirements, answerability):
        requirement_id = str(specification.get("requirement_id"))
        expected = specification.get("expected_value")
        value_type = specification.get("value_type")
        if value_type == "number" and expected is not None:
            if not _numeric_role_present(
                reference,
                value=expected,
                specification=specification,
            ):
                reasons.append("numeric_role_binding_failed")
        elif value_type == "set" or specification.get("expected_members") is not None:
            expected_members = specification.get("expected_members", [])
            missing = [
                item for item in expected_members
                if _normalized(item) not in normalized_reference
            ]
            if missing:
                reasons.append("missing_required_members")
        elif (
            expected is not None
            and (
                value_type in {"exact_phrase", "identifier"}
                or specification.get("match_policy") in {"exact", "exact_phrase", "canonical"}
                or requirements.get("answer_structure") == "exact_phrase"
            )
        ):
            if _normalized(expected) not in normalized_reference:
                reasons.append("incomplete_exact_phrase")
        result = qualification_results.get(requirement_id, {})
        if result.get("relation_matches") is False:
            reasons.append("numeric_role_binding_failed")
        if result.get("qualifiers_present") is False:
            reasons.append("missing_mandatory_qualifier")
        for qualifier in specification.get("required_qualifiers", []):
            if isinstance(qualifier, str):
                continue
            if not isinstance(qualifier, dict):
                continue
            if qualifier.get("match_policy") not in {"exact", "exact_phrase", "canonical"}:
                continue
            canonical = qualifier.get("canonical_text") or qualifier.get("value")
            if canonical and _normalized(canonical) not in normalized_reference:
                reasons.append("missing_mandatory_qualifier")

    if qualification.get("numeric_role_binding") is False:
        reasons.append("numeric_role_binding_failed")
    if qualification.get("logical_relation_matches") is False:
        reasons.append("numeric_role_binding_failed")
    if qualification.get("premise_restatement") is True:
        reasons.append("premise_restatement")
    if qualification.get("missing_required_members"):
        reasons.append("missing_required_members")
    if qualification.get("unsupported_members"):
        reasons.append("unsupported_members")
    if qualification.get("exact_phrase_complete") is False:
        reasons.append("incomplete_exact_phrase")
    if qualification.get("mandatory_qualifiers_present") is False:
        reasons.append("missing_mandatory_qualifier")
    inference_policy = answerability.get(
        "allowed_inference_policy",
        requirements.get("allowed_inference_policy", "source_explicit_only"),
    )
    if qualification.get("unsupported_claims") and (
        inference_policy == "source_explicit_only"
        or qualification.get("unsupported_claims_material", True)
    ):
        reasons.append("unsupported_claim")
    if qualification.get("unsupported_reference_claims"):
        reasons.append("unsupported_reference_claim")
    if qualification.get("contradicted_claims"):
        reasons.append("contradicted_claim")
    return list(dict.fromkeys(reasons))


def qualification_decision(
    requirements: dict[str, Any] | None,
    answerability: dict[str, Any] | None,
    qualification: dict[str, Any] | None,
    reference: str,
    *,
    infrastructure_uncertainty: bool = False,
    provenance_resolvable: bool = True,
) -> dict[str, Any]:
    if infrastructure_uncertainty or requirements is None or answerability is None or qualification is None:
        return {
            "schema": QUALIFICATION_DECISION_SCHEMA,
            "decision": "needs_review",
            "reason_codes": ["qualification_infrastructure_uncertainty"],
            "quality_label": "not_determined",
            "question_validity": "not_determined",
            "answerable_at_requested_specificity": None,
            "reference_correctness": "not_assessable",
            "reference_completeness": "not_assessable",
            "source_faithfulness": "not_assessable",
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
            "deterministic_gate_results": {},
            "qualification_infrastructure_status": "uncertain",
            "operational_acceptance": "needs_review",
            "rewrite_allowed": False,
        }

    reasons: list[str] = []
    question_validity = str(requirements.get("question_validity", "valid"))
    if question_validity == "invalid":
        reasons.append("question_invalid")
    elif question_validity == "uncertain":
        reasons.append("question_validity_uncertain")
    answerable = bool(answerability.get("answerable_at_requested_specificity", False))
    if not answerable:
        reasons.append("source_not_answerable_at_requested_specificity")
    answers_question = bool(qualification.get("answers_question", False))
    if not answers_question:
        reasons.append("reference_does_not_answer_question")
    coverage = float(qualification.get("required_slot_coverage", 0.0))
    if answerable and coverage < 1.0:
        reasons.append("missing_required_facts")
    deterministic_failures = deterministic_reference_checks(
        requirements,
        answerability,
        qualification,
        reference,
        provenance_resolvable=provenance_resolvable,
    )
    reasons.extend(deterministic_failures)
    reasons = list(dict.fromkeys(reasons))
    supported_claims = list(qualification.get("supported_claims", []))
    unsupported_claims = list(qualification.get("unsupported_claims", []))
    contradicted_claims = list(qualification.get("contradicted_claims", []))
    reference_correctness = qualification.get("reference_correctness")
    if reference_correctness is None:
        if not answerable:
            reference_correctness = "not_assessable"
        elif contradicted_claims:
            reference_correctness = "incorrect"
        elif answers_question and coverage >= 1.0:
            reference_correctness = "correct"
        elif answers_question and coverage > 0.0:
            reference_correctness = "partially_correct"
        else:
            reference_correctness = "incorrect"
    reference_completeness = qualification.get("reference_completeness")
    if reference_completeness is None:
        if not answerable:
            reference_completeness = "not_assessable"
        else:
            reference_completeness = "complete" if coverage >= 1.0 else "incomplete"
    source_faithfulness = qualification.get("source_faithfulness")
    if unsupported_claims or contradicted_claims:
        source_faithfulness = "failed"
    elif source_faithfulness is None:
        source_faithfulness = "passed"
    if (
        source_faithfulness == "failed"
        and not unsupported_claims
        and not contradicted_claims
    ):
        reasons.append("source_faithfulness_failed")
    decision = (
        "needs_review"
        if question_validity == "uncertain"
        else "rejected" if reasons else "accepted"
    )
    quality_label = (
        "qualified"
        if decision == "accepted"
        else "not_determined" if decision == "needs_review" else str(reference_correctness)
    )
    gate_results = {
        "question_valid": question_validity == "valid",
        "source_answerable_at_requested_specificity": answerable,
        "reference_answers_question": answers_question,
        "required_facts_complete": coverage >= 1.0,
        "source_faithful": source_faithfulness == "passed",
        "provenance_resolvable": provenance_resolvable,
        "deterministic_failures": deterministic_failures,
    }
    return {
        "schema": QUALIFICATION_DECISION_SCHEMA,
        "decision": decision,
        "reason_codes": reasons,
        "quality_label": quality_label,
        "question_validity": question_validity,
        "answerable_at_requested_specificity": answerable,
        "reference_correctness": reference_correctness,
        "reference_completeness": reference_completeness,
        "source_faithfulness": source_faithfulness,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "contradicted_claims": contradicted_claims,
        "deterministic_gate_results": gate_results,
        "qualification_infrastructure_status": "reliable",
        "operational_acceptance": decision,
        "rewrite_allowed": False,
    }


@dataclass(frozen=True, slots=True)
class QualificationResult:
    answer_requirements: dict[str, Any] | None
    source_answerability: dict[str, Any] | None
    reference_qualification: dict[str, Any] | None
    decision: dict[str, Any]
    raw_attempts: dict[str, list[str]]


class QualificationPipeline:
    """Run candidate-blind semantic stages and a deterministic final decision."""

    def __init__(
        self,
        *,
        pool: Any,
        role_configs: dict[str, GeneratorConfig],
        context_limit: int | None,
        prompt_versions: dict[str, str],
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("qualification max_attempts must be at least 1")
        self.adapters = {name: StructuredAdapter(pool, config) for name, config in role_configs.items()}
        self.context_limit = context_limit
        self.prompt_versions = prompt_versions
        self.max_attempts = max_attempts
        self.prompt_dir = Path(__file__).with_name("prompts")

    def _stage(
        self,
        name: str,
        payload: dict[str, Any],
        parser: Callable[[str], dict[str, Any]],
        *,
        evidence_ids: list[str],
        calls: list[dict[str, Any]],
        raw_attempts: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        prompt = self.prompt_dir.joinpath(f"qualification_{name}.txt").read_text(encoding="utf-8")
        for _ in range(self.max_attempts):
            try:
                raw = self.adapters[name].ask_budgeted(
                    prompt + "\n\nINPUT:\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False),
                    "Return one strict JSON object and no prose.",
                    context_limit=self.context_limit,
                    role=name,
                    prompt_version=self.prompt_versions.get(name, "1.0"),
                    evidence_ids=evidence_ids,
                    calls=calls,
                )
                raw_attempts.setdefault(name, []).append(raw)
                return parser(raw)
            except (json.JSONDecodeError, QualificationOutputError, RuntimeError, ValueError) as exc:
                raw_attempts.setdefault(name, []).append(f"ERROR: {type(exc).__name__}: {exc}")
        return None

    def qualify(
        self,
        *,
        question: str,
        reference: str,
        source_text: str,
        provenance: dict[str, Any],
        evidence_ids: list[str],
        calls: list[dict[str, Any]],
        provenance_resolvable: bool = True,
    ) -> QualificationResult:
        attempts: dict[str, list[str]] = {}
        requirements = self._stage(
            "answer_requirements",
            {"question": question},
            _requirements,
            evidence_ids=[],
            calls=calls,
            raw_attempts=attempts,
        )
        if requirements is None:
            decision = qualification_decision(None, None, None, reference, infrastructure_uncertainty=True)
            return QualificationResult(None, None, None, decision, attempts)

        answerability = self._stage(
            "source_answerability",
            {
                "answer_requirements": requirements,
                "source_evidence": source_text,
                "provenance": provenance,
            },
            _answerability,
            evidence_ids=evidence_ids,
            calls=calls,
            raw_attempts=attempts,
        )
        if answerability is None:
            decision = qualification_decision(requirements, None, None, reference, infrastructure_uncertainty=True)
            return QualificationResult(requirements, None, None, decision, attempts)
        reference_result = self._stage(
            "reference_qualification",
            {
                "answer_requirements": requirements,
                "source_answerability": answerability,
                "generated_reference": reference,
            },
            _reference_qualification,
            evidence_ids=evidence_ids,
            calls=calls,
            raw_attempts=attempts,
        )
        decision = qualification_decision(
            requirements,
            answerability,
            reference_result,
            reference,
            infrastructure_uncertainty=reference_result is None,
            provenance_resolvable=provenance_resolvable,
        )
        return QualificationResult(requirements, answerability, reference_result, decision, attempts)
