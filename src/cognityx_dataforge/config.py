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


@dataclass(frozen=True, slots=True)
class DataForgeConfig:
    generator: GeneratorModelConfig
    prompt_version: str = "v0.1"

    @classmethod
    def load(cls, path: str | Path) -> "DataForgeConfig":
        payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        generator = payload["models"]["generator"]
        return cls(
            generator=GeneratorModelConfig(
                model=str(generator["model"]),
                backend=str(generator["backend"]),
                profile=str(generator["profile"]),
                max_output_tokens=int(generator["max_output_tokens"]),
            ),
            prompt_version=str(payload.get("prompt_version", "v0.1")),
        )

