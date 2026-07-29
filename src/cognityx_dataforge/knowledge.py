from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from cognityx_dataforge.dataset import deterministic_id


KNOWLEDGE_UNIT_SCHEMA_VERSION = "cognityx.dataforge.knowledge-unit/v1"


@dataclass(frozen=True, slots=True)
class KnowledgeUnit:
    knowledge_unit_id: str
    source_evidence_ids: tuple[str, ...]
    canonical_statement: str
    supporting_facts: tuple[str, ...]
    concepts: tuple[str, ...]
    prerequisites: tuple[str, ...]
    difficulty: str
    ambiguity_flags: tuple[str, ...]
    generator_model: str
    prompt_version: str
    schema_version: str = KNOWLEDGE_UNIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("source_evidence_ids", "supporting_facts", "concepts", "prerequisites", "ambiguity_flags"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KnowledgeUnit":
        return cls(
            knowledge_unit_id=str(value["knowledge_unit_id"]),
            source_evidence_ids=tuple(value.get("source_evidence_ids", [])),
            canonical_statement=str(value["canonical_statement"]),
            supporting_facts=tuple(value.get("supporting_facts", [])),
            concepts=tuple(value.get("concepts", [])),
            prerequisites=tuple(value.get("prerequisites", [])),
            difficulty=str(value.get("difficulty", "unknown")),
            ambiguity_flags=tuple(value.get("ambiguity_flags", [])),
            generator_model=str(value["generator_model"]),
            prompt_version=str(value["prompt_version"]),
            schema_version=str(value.get("schema_version", KNOWLEDGE_UNIT_SCHEMA_VERSION)),
        )


def parse_knowledge_units(payload: str, evidence_id: str, model: str, prompt_version: str) -> tuple[KnowledgeUnit, ...]:
    data = json.loads(payload)
    raw_units = data.get("knowledge_units", data) if isinstance(data, dict) else data
    if not isinstance(raw_units, list):
        raise ValueError("Knowledge-unit discovery must return a JSON list")
    units: list[KnowledgeUnit] = []
    for raw in raw_units:
        if not isinstance(raw, dict) or not str(raw.get("canonical_statement", "")).strip():
            raise ValueError("Knowledge unit lacks canonical_statement")
        evidence_ids = tuple(str(item) for item in raw.get("source_evidence_ids", [evidence_id]))
        if evidence_id not in evidence_ids:
            raise ValueError("Knowledge unit must retain its source evidence id")
        statement = str(raw["canonical_statement"]).strip()
        units.append(KnowledgeUnit(
            knowledge_unit_id=deterministic_id(evidence_id, statement),
            source_evidence_ids=evidence_ids,
            canonical_statement=statement,
            supporting_facts=tuple(str(item) for item in raw.get("supporting_facts", [])),
            concepts=tuple(str(item) for item in raw.get("concepts", [])),
            prerequisites=tuple(str(item) for item in raw.get("prerequisites", [])),
            difficulty=str(raw.get("difficulty", "unknown")),
            ambiguity_flags=tuple(str(item) for item in raw.get("ambiguity_flags", [])),
            generator_model=model,
            prompt_version=prompt_version,
        ))
    return tuple(units)
