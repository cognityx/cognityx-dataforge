from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class GeneratorModelConfig:
    model: str
    provider: str = "local"
    backend: str | None = "vllm"
    profile: str | None = "bf16"
    server_profile: str | None = None
    max_output_tokens: int | None = None


ValidatorModelConfig = GeneratorModelConfig


@dataclass(frozen=True, slots=True)
class DataForgeConfig:
    generator: GeneratorModelConfig
    knowledge_unit: GeneratorModelConfig | None = None
    qa_generator: GeneratorModelConfig | None = None
    validator: ValidatorModelConfig | None = None
    prompt_versions: dict[str, str] = None  # type: ignore[assignment]
    context_limit_tokens: int | None = None
    base_url: str | None = None
    manager_url: str | None = None
    auto_start_local: bool = False
    startup_timeout_seconds: float = 600.0
    commercial_enabled: bool = False

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
        models = payload.get("models", {})
        generator = models["generator"]
        def role_config(role_payload):
            if not role_payload:
                return None
            return GeneratorModelConfig(
                model=str(role_payload["model"]), provider=str(role_payload.get("provider", "local")),
                backend=(str(role_payload["backend"]) if role_payload.get("backend") is not None else None),
                profile=(str(role_payload["profile"]) if role_payload.get("profile") is not None else None),
                server_profile=(str(role_payload["server_profile"]) if role_payload.get("server_profile") is not None else None),
                max_output_tokens=(int(role_payload["max_output_tokens"]) if role_payload.get("max_output_tokens") is not None else None),
            )
        knowledge_unit = role_config(models.get("knowledge_unit") or generator)
        qa_generator = role_config(models.get("qa_generator") or generator)
        validator_payload = models.get("validator")
        validator = None
        if validator_payload:
            validator = role_config(validator_payload)
        prompt_versions = payload.get("prompt_versions")
        if not prompt_versions:
            prompt_versions = {"generation": str(payload.get("prompt_version", "1.0")), "knowledge_unit": "1.0", "validation": "1.0"}
        return cls(
            generator=GeneratorModelConfig(
                model=str(generator["model"]),
                provider=str(generator.get("provider", "local")),
                backend=(str(generator["backend"]) if generator.get("backend") is not None else None),
                profile=(str(generator["profile"]) if generator.get("profile") is not None else None),
                server_profile=(str(generator["server_profile"]) if generator.get("server_profile") is not None else None),
                max_output_tokens=(int(generator["max_output_tokens"]) if generator.get("max_output_tokens") is not None else None),
            ),
            knowledge_unit=knowledge_unit,
            qa_generator=qa_generator,
            validator=validator,
            prompt_versions={str(key): str(value) for key, value in prompt_versions.items()},
            context_limit_tokens=(int(payload.get("context_limit_tokens")) if payload.get("context_limit_tokens") is not None else None),
            base_url=(str(payload["inference"]["base_url"]) if payload.get("inference", {}).get("base_url") is not None else None),
            manager_url=(str(payload["inference"]["manager_url"]) if payload.get("inference", {}).get("manager_url") is not None else None),
            auto_start_local=bool(payload.get("inference", {}).get("auto_start_local", False)),
            startup_timeout_seconds=float(payload.get("inference", {}).get("startup_timeout_seconds", 600)),
            commercial_enabled=bool(payload.get("commercial", {}).get("enabled", False)),
        )
