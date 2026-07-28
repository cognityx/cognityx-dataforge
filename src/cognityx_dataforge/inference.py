from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    model: str
    backend: str
    profile: str
    max_output_tokens: int


class GeneratorAdapter:
    def __init__(self, client: Any, config: GeneratorConfig) -> None:
        self.client = client
        self.config = config

    def count_tokens(self, text: str) -> int | None:
        try:
            result = self.client.count_input_tokens(
                model=self.config.model,
                backend=self.config.backend,
                profile=self.config.profile,
                messages=[{"role": "user", "content": text}],
            )
        except Exception:
            return None
        return int(result.get("input_tokens")) if result and result.get("input_tokens") is not None else None

    def generate(self, prompt: str) -> dict[str, str]:
        response = self.client.chat(
            model=self.config.model,
            backend=self.config.backend,
            profile=self.config.profile,
            messages=[
                {"role": "system", "content": "Return strict JSON with keys instruction and answer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_output_tokens,
        )
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Malformed generator output")
        instruction = str(data.get("instruction", "")).strip()
        answer = str(data.get("answer", "")).strip()
        if not instruction or not answer:
            raise ValueError("Incomplete generator output")
        return {"instruction": instruction, "answer": answer}
