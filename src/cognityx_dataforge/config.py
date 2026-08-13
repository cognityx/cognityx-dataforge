from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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
    probe_generator: GeneratorModelConfig | None = None
    student: GeneratorModelConfig | None = None
    probe_judge: GeneratorModelConfig | None = None
    answer_requirements: GeneratorModelConfig | None = None
    source_answerability: GeneratorModelConfig | None = None
    reference_qualification: GeneratorModelConfig | None = None
    prompt_versions: dict[str, str] = None  # type: ignore[assignment]
    context_limit_tokens: int | None = None
    base_url: str | None = None
    manager_url: str | None = None
    auto_start_local: bool = False
    startup_timeout_seconds: float = 600.0
    commercial_enabled: bool = False
    external_enabled: bool = False
    allowed_providers: tuple[str, ...] = ()
    data_classification: str = "internal"
    permit_external_sensitive_data: bool = False
    probes_per_unit: int = 2
    include_classes: tuple[str, ...] = ("partial", "unknown")
    known_sample_rate: float = 0.0
    qualification_max_attempts: int = 2
    split_seed: str = "dataforge-v1"

    def __post_init__(self) -> None:
        if self.prompt_versions is None:
            object.__setattr__(self, "prompt_versions", {"generation": "1.0", "knowledge_unit": "1.0", "validation": "1.0", "probe_generation": "2.0", "student_probe": "2.0", "probe_judgment": "2.0", "probed_qa_generation": "2.0", "probed_qa_validation": "2.0", "answer_requirements": "1.0", "source_answerability": "1.0", "reference_qualification": "1.0"})
        if self.qualification_max_attempts < 1:
            raise ValueError("qualification.max_attempts must be at least 1")

    @property
    def prompt_version(self) -> str:
        return self.prompt_versions.get("generation", "1.0")

    @classmethod
    def load(cls, path: str | Path) -> DataForgeConfig:
        return resolve_dataforge_config(path).configuration

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> DataForgeConfig:
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
        probe_generator = role_config(models.get("probe_generator") or models.get("qa_generator") or generator)
        student = role_config(models.get("student") or generator)
        probe_judge = role_config(models.get("probe_judge") or models.get("validator") or generator)
        answer_requirements = role_config(models.get("answer_requirements") or models.get("validator") or generator)
        source_answerability = role_config(models.get("source_answerability") or models.get("validator") or generator)
        reference_qualification = role_config(models.get("reference_qualification") or models.get("validator") or generator)
        probing = payload.get("probing", {})
        prompt_versions = payload.get("prompt_versions")
        if not prompt_versions:
            prompt_versions = {"generation": str(payload.get("prompt_version", "1.0")), "knowledge_unit": "1.0", "validation": "1.0"}
        prompt_versions = dict(prompt_versions)
        prompt_versions.setdefault("probe_generation", "2.0")
        prompt_versions.setdefault("student_probe", "2.0")
        prompt_versions.setdefault("probe_judgment", "2.0")
        prompt_versions.setdefault("probed_qa_generation", "2.0")
        prompt_versions.setdefault("probed_qa_validation", "2.0")
        prompt_versions.setdefault("answer_requirements", "1.0")
        prompt_versions.setdefault("source_answerability", "1.0")
        prompt_versions.setdefault("reference_qualification", "1.0")
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
            probe_generator=probe_generator,
            student=student,
            probe_judge=probe_judge,
            answer_requirements=answer_requirements,
            source_answerability=source_answerability,
            reference_qualification=reference_qualification,
            prompt_versions={str(key): str(value) for key, value in prompt_versions.items()},
            context_limit_tokens=(int(payload.get("context_limit_tokens")) if payload.get("context_limit_tokens") is not None else None),
            base_url=(str(payload["inference"]["base_url"]) if payload.get("inference", {}).get("base_url") is not None else None),
            manager_url=(str(payload["inference"]["manager_url"]) if payload.get("inference", {}).get("manager_url") is not None else None),
            auto_start_local=bool(payload.get("inference", {}).get("auto_start_local", False)),
            startup_timeout_seconds=float(payload.get("inference", {}).get("startup_timeout_seconds", 600)),
            commercial_enabled=bool(payload.get("commercial", {}).get("enabled", False)),
            external_enabled=bool(payload.get("external_inference", {}).get("enabled", False)) or bool(payload.get("commercial", {}).get("enabled", False)),
            allowed_providers=tuple(str(item) for item in payload.get("external_inference", {}).get("allowed_providers", ())),
            data_classification=str(payload.get("data", {}).get("classification", "internal")),
            permit_external_sensitive_data=bool(payload.get("data", {}).get("permit_external_sensitive_data", False)),
            probes_per_unit=int(probing.get("probes_per_unit", 2)),
            include_classes=tuple(str(item) for item in probing.get("include_classes", ("partial", "unknown"))),
            known_sample_rate=float(probing.get("known_sample_rate", 0.0)),
            qualification_max_attempts=int(payload.get("qualification", {}).get("max_attempts", 2)),
            split_seed=str(payload.get("splitting", {}).get("seed", "dataforge-v1")),
        )


@dataclass(frozen=True, slots=True)
class DataForgeConfigResolution:
    configuration: DataForgeConfig
    path: Path
    file_sha256: str
    changed_keys: tuple[str, ...]
    field_sources: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        effective = _safe_value(asdict(self.configuration))
        return {
            "component": "dataforge",
            "configuration_kind": "scientific-workload",
            "valid": True,
            "master_config": {
                "kind": "file",
                "path": str(self.path),
                "selected_by": "explicit",
                "sha256": self.file_sha256,
            },
            "config_layers": [{
                "path": str(self.path),
                "selected_by": "explicit",
                "sha256": self.file_sha256,
                "changed_keys": list(self.changed_keys),
            }],
            "field_sources": dict(sorted(self.field_sources.items())),
            "overrides": [],
            "effective": effective,
            "warnings": [],
            "errors": [],
        }


def resolve_dataforge_config(path: str | Path) -> DataForgeConfigResolution:
    selected = Path(path).expanduser().resolve()
    raw = selected.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    configuration = DataForgeConfig._from_payload(payload)
    return DataForgeConfigResolution(
        configuration=configuration,
        path=selected,
        file_sha256=sha256(raw).hexdigest(),
        changed_keys=tuple(sorted(_flatten_keys(payload))),
        field_sources=_dataforge_field_sources(payload, configuration, selected),
    )


def _flatten_keys(value: object, prefix: str = "") -> tuple[str, ...]:
    if not isinstance(value, dict):
        return (prefix,) if prefix else ()
    keys: list[str] = []
    for name, item in value.items():
        dotted = f"{prefix}.{name}" if prefix else str(name)
        nested = _flatten_keys(item, dotted)
        keys.extend(nested or (dotted,))
    return tuple(keys)


def _safe_value(value: object, key: str = "") -> object:
    lowered = key.lower()
    if any(marker in lowered for marker in ("secret", "password", "token", "api_key")):
        return "<redacted>" if value is not None else None
    if isinstance(value, dict):
        return {str(name): _safe_value(item, str(name)) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_safe_value(item, key) for item in value]
    if isinstance(value, str) and "://" in value:
        return _redacted_uri(value)
    return value


def _dataforge_field_sources(
    payload: Mapping[str, Any],
    configuration: DataForgeConfig,
    path: Path,
) -> Mapping[str, str]:
    effective_keys = _flatten_keys(asdict(configuration))
    sources = {key: "built-in" for key in effective_keys}
    file_source = str(path)
    models = payload.get("models") or {}
    role_sources = {
        "generator": "generator",
        "knowledge_unit": "knowledge_unit" if models.get("knowledge_unit") else "generator",
        "qa_generator": "qa_generator" if models.get("qa_generator") else "generator",
        "validator": "validator" if models.get("validator") else None,
        "probe_generator": next(
            (name for name in ("probe_generator", "qa_generator", "generator") if models.get(name)),
            None,
        ),
        "student": "student" if models.get("student") else "generator",
        "probe_judge": next(
            (name for name in ("probe_judge", "validator", "generator") if models.get(name)),
            None,
        ),
        "answer_requirements": next(
            (name for name in ("answer_requirements", "validator", "generator") if models.get(name)),
            None,
        ),
        "source_answerability": next(
            (name for name in ("source_answerability", "validator", "generator") if models.get(name)),
            None,
        ),
        "reference_qualification": next(
            (name for name in ("reference_qualification", "validator", "generator") if models.get(name)),
            None,
        ),
    }
    for effective_role, input_role in role_sources.items():
        if input_role is None:
            continue
        raw_role = models.get(input_role) or {}
        for name in raw_role:
            dotted = f"{effective_role}.{name}"
            if dotted in sources:
                sources[dotted] = file_source

    prompt_versions = payload.get("prompt_versions") or {}
    for name in prompt_versions:
        sources[f"prompt_versions.{name}"] = file_source
    if "prompt_version" in payload:
        sources["prompt_versions.generation"] = file_source

    direct = {
        "context_limit_tokens": (payload, "context_limit_tokens"),
        "base_url": (payload.get("inference") or {}, "base_url"),
        "manager_url": (payload.get("inference") or {}, "manager_url"),
        "auto_start_local": (payload.get("inference") or {}, "auto_start_local"),
        "startup_timeout_seconds": (
            payload.get("inference") or {},
            "startup_timeout_seconds",
        ),
        "commercial_enabled": (payload.get("commercial") or {}, "enabled"),
        "allowed_providers": (
            payload.get("external_inference") or {},
            "allowed_providers",
        ),
        "data_classification": (payload.get("data") or {}, "classification"),
        "permit_external_sensitive_data": (
            payload.get("data") or {},
            "permit_external_sensitive_data",
        ),
        "probes_per_unit": (payload.get("probing") or {}, "probes_per_unit"),
        "include_classes": (payload.get("probing") or {}, "include_classes"),
        "known_sample_rate": (payload.get("probing") or {}, "known_sample_rate"),
        "qualification_max_attempts": (
            payload.get("qualification") or {},
            "max_attempts",
        ),
        "split_seed": (payload.get("splitting") or {}, "seed"),
    }
    for effective_name, (section, input_name) in direct.items():
        if input_name in section:
            sources[effective_name] = file_source
    if "enabled" in (payload.get("external_inference") or {}) or "enabled" in (
        payload.get("commercial") or {}
    ):
        sources["external_enabled"] = file_source
    return sources


def _redacted_uri(value: str) -> str:
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    netloc = (
        host
        if parsed.username is not None or parsed.password is not None
        else parsed.netloc
    )
    query = urlencode([
        (
            name,
            "<redacted>"
            if any(
                marker in name.lower()
                for marker in ("secret", "password", "token", "api_key")
            )
            else item,
        )
        for name, item in parse_qsl(parsed.query, keep_blank_values=True)
    ])
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
