from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class GeneratorModelConfig:
    model: str
    backend: str
    profile: str
    max_output_tokens: int


ValidatorModelConfig = GeneratorModelConfig


@dataclass(frozen=True, slots=True)
class DataForgeConfig:
    generator: GeneratorModelConfig
    validator: ValidatorModelConfig | None = None
    prompt_versions: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.prompt_versions is None:
            object.__setattr__(self, "prompt_versions", {"generation": "1.0", "knowledge_unit": "1.0", "validation": "1.0"})

    @property
    def prompt_version(self) -> str:
        return self.prompt_versions.get("generation", "1.0")

    @classmethod
    def load(cls, path: str | Path) -> "DataForgeConfig":
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        generator = payload["models"]["generator"]
        validator_payload = payload.get("models", {}).get("validator")
        validator = None
        if validator_payload:
            validator = GeneratorModelConfig(
                model=str(validator_payload["model"]),
                backend=str(validator_payload["backend"]),
                profile=str(validator_payload["profile"]),
                max_output_tokens=int(validator_payload["max_output_tokens"]),
            )
        prompt_versions = payload.get("prompt_versions")
        if not prompt_versions:
            prompt_versions = {"generation": str(payload.get("prompt_version", "1.0")), "knowledge_unit": "1.0", "validation": "1.0"}
        return cls(
            generator=GeneratorModelConfig(
                model=str(generator["model"]),
                backend=str(generator["backend"]),
                profile=str(generator["profile"]),
                max_output_tokens=int(generator["max_output_tokens"]),
            ),
            validator=validator,
            prompt_versions={str(key): str(value) for key, value in prompt_versions.items()},
        )
