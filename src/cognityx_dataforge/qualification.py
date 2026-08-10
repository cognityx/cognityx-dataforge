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
    return {"schema": ANSWER_REQUIREMENTS_SCHEMA, **value}


def _answerability(raw: str) -> dict[str, Any]:
    value = _json_object(raw)
    if not isinstance(value.get("answerable_at_requested_specificity"), bool):
        raise QualificationOutputError("source answerability needs a boolean answerable_at_requested_specificity")
    if not isinstance(value.get("slot_values", {}), dict):
        raise QualificationOutputError("source answerability slot_values must be an object")
    return {"schema": SOURCE_ANSWERABILITY_SCHEMA, **value}


def _reference_qualification(raw: str) -> dict[str, Any]:
    value = _json_object(raw)
    if not isinstance(value.get("answers_question"), bool):
        raise QualificationOutputError("reference qualification needs a boolean answers_question")
    coverage = value.get("required_slot_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
        raise QualificationOutputError("reference qualification needs numeric required_slot_coverage")
    if not 0.0 <= float(coverage) <= 1.0:
        raise QualificationOutputError("required_slot_coverage must be between 0 and 1")
    return {"schema": REFERENCE_QUALIFICATION_SCHEMA, **value}


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


def _numeric_role_present(reference: str, *, value: Any, role: str, unit: str | None) -> bool:
    normalized = _normalized(reference)
    number = re.escape(_normalized(value))
    role_terms = {
        "sustained_wind_stop": "sustained",
        "gust_stop": "gust",
        "paid_hours_minimum": "paid hours",
    }
    role_term = role_terms.get(role, _normalized(role).replace(" stop", ""))
    unit_term = _normalized(unit) if unit else ""
    windows = (
        rf"{number}(?:\s+{re.escape(unit_term)})?\s+(?:\w+\s+){{0,3}}{re.escape(role_term)}",
        rf"{re.escape(role_term)}(?:\s+\w+){{0,3}}\s+{number}(?:\s+{re.escape(unit_term)})?",
    )
    return any(re.search(pattern, normalized) for pattern in windows)


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

    slot_values = answerability.get("slot_values", {})
    normalized_reference = _normalized(reference)
    for slot in requirements.get("required_slots", []):
        expected = slot_values.get(slot)
        if isinstance(expected, dict) and "value" in expected:
            if not _numeric_role_present(
                reference,
                value=expected["value"],
                role=str(slot),
                unit=str(expected.get("unit")) if expected.get("unit") else None,
            ):
                reasons.append("numeric_role_binding_failed")
        elif isinstance(expected, list):
            missing = [item for item in expected if _normalized(item) not in normalized_reference]
            if missing:
                reasons.append("missing_required_members")
        elif expected is not None and requirements.get("answer_structure") == "exact_phrase":
            if _normalized(expected) not in normalized_reference:
                reasons.append("incomplete_exact_phrase")

    if slot_values.get("logical_relation") == "OR" and " or " not in f" {normalized_reference} ":
        reasons.append("numeric_role_binding_failed")
    if "exactly" in _normalized(answerability.get("source_text", "")) and "exactly" not in normalized_reference:
        reasons.append("missing_mandatory_qualifier")
    if "or higher" in normalized_reference and "or higher" not in _normalized(answerability.get("source_text", "")):
        reasons.append("unsupported_claim")

    if qualification.get("numeric_role_binding") is False:
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
    if qualification.get("unsupported_claims"):
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
            "rewrite_allowed": False,
        }

    reasons: list[str] = []
    if not answerability.get("answerable_at_requested_specificity", False):
        reasons.append("source_not_answerable_at_requested_specificity")
    if not qualification.get("answers_question", False):
        reasons.append("reference_does_not_answer_question")
    if answerability.get("answerable_at_requested_specificity", False) and float(qualification.get("required_slot_coverage", 0.0)) < 1.0:
        reasons.append("missing_required_facts")
    reasons.extend(deterministic_reference_checks(
        requirements,
        answerability,
        qualification,
        reference,
        provenance_resolvable=provenance_resolvable,
    ))
    reasons = list(dict.fromkeys(reasons))
    return {
        "schema": QUALIFICATION_DECISION_SCHEMA,
        "decision": "rejected" if reasons else "accepted",
        "reason_codes": reasons,
        "quality_label": "incorrect" if reasons else "qualified",
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
        answerability = {**answerability, "source_text": source_text}

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
