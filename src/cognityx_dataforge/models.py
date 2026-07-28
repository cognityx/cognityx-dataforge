from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    evidence_id: str
    document_id: str
    source_asset_id: str | None
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    record_id: str
    messages: tuple[dict[str, str], dict[str, str]]
    split: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "messages": list(self.messages),
            "split": self.split,
            "metadata": self.metadata,
        }

