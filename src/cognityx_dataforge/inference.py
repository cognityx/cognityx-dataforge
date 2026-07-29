from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class TokenBudgetError(ValueError):
    def __init__(self, message: str, *, input_tokens: int, max_output_tokens: int, context_limit: int):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.max_output_tokens = max_output_tokens
        self.context_limit = context_limit


def normalize_input_token_count(value: int | None) -> int | None:
    """Normalize the Cognityx inference contract without accepting mappings."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("count_input_tokens() must return int or None")
    return value


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


class StructuredAdapter:
    def __init__(self, client: Any, config: GeneratorConfig) -> None:
        self.client = client
        self.config = config

    def ask(self, prompt: str, system: str) -> str:
        response = self.client.chat(
            model=self.config.model,
            backend=self.config.backend,
            profile=self.config.profile,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            max_tokens=self.config.max_output_tokens,
        )
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")

    def ask_budgeted(self, prompt: str, system: str, *, context_limit: int | None, role: str, prompt_version: str, evidence_ids: list[str], calls: list[dict[str, Any]]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        counter = getattr(self.client, "count_input_tokens", None)
        if counter is None:
            raise RuntimeError("Inference client must provide count_input_tokens for budgeted calls")
        result = normalize_input_token_count(counter(model=self.config.model, backend=self.config.backend, profile=self.config.profile, messages=messages))
        if result is None:
            raise RuntimeError("Inference client could not count input tokens")
        input_tokens = result
        limit = context_limit
        if not limit:
            raise RuntimeError("Configure context_limit_tokens or provide a certified context limit")
        if input_tokens + self.config.max_output_tokens > limit:
            raise TokenBudgetError("Model request exceeds context budget", input_tokens=input_tokens, max_output_tokens=self.config.max_output_tokens, context_limit=limit)
        calls.append({"role": role, "model": self.config.model, "backend": self.config.backend, "profile": self.config.profile, "history_mode": "none", "input_tokens": input_tokens, "max_output_tokens": self.config.max_output_tokens, "prompt_version": prompt_version, "evidence_ids": list(evidence_ids)})
        return self.ask(prompt, system)


def load_inference_client() -> Any:
    try:
        from cognityx_inference.client import CognityxInferenceClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Inference support is not installed. Install DataForge with "
            "`pip install cognityx-dataforge[inference]` or inject an inference client."
        ) from exc
    return CognityxInferenceClient()
