from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cognityx_dataforge.config import GeneratorModelConfig


class TokenBudgetError(ValueError):
    def __init__(self, message: str, *, input_tokens: int | None = None, max_output_tokens: int | None = None, context_limit: int | None = None):
        super().__init__(message)
        self.input_tokens = input_tokens
        self.max_output_tokens = max_output_tokens
        self.context_limit = context_limit


def normalize_input_token_count(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("count_input_tokens() must return int or None")
    return value


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    model: str
    provider: str = "local"
    backend: str | None = "vllm"
    profile: str | None = "bf16"
    server_profile: str | None = None
    max_output_tokens: int | None = None


def _content(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content)
    return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))


def response_metadata(response: Any, config: GeneratorConfig, role: str) -> dict[str, Any]:
    value = response.to_dict() if hasattr(response, "to_dict") else dict(response)
    cognityx = value.get("cognityx", {}) or {}
    return {
        "request_id": value.get("id") or cognityx.get("request_id"),
        "role": role,
        "provider": config.provider,
        "model": config.model,
        "backend": config.backend if config.provider == "local" else None,
        "profile": config.profile if config.provider == "local" else None,
        "server_profile": config.server_profile if config.provider == "local" else None,
        "token_budget": cognityx.get("token_budget"),
        "usage": value.get("usage") or cognityx.get("usage"),
        "timings": cognityx.get("timings"),
        "finish_reason": (value.get("choices", [{}])[0].get("finish_reason") if isinstance(value, dict) else None) or cognityx.get("finish_reason"),
        "rate_limit_diagnostics": (cognityx.get("extensions", {}) or {}).get("rate_limits", {}),
        "history_mode": "none",
    }


class InferenceClientPool:
    def __init__(self, *, config: Any, injected_client: Any | None = None) -> None:
        self.config = config
        self.injected_client = injected_client
        self._clients: dict[tuple[Any, ...], Any] = {}
        self._preflight: dict[tuple[Any, ...], dict[str, Any]] = {}

    def client_for(self, role: GeneratorConfig) -> Any:
        if role.provider != "local":
            classification = getattr(self.config, "data_classification", "internal")
            if classification in {"sensitive", "restricted", "confidential"} and not getattr(self.config, "permit_external_sensitive_data", False):
                raise RuntimeError(f"External provider '{role.provider}' is blocked for {classification} data")
            allowed = set(getattr(self.config, "allowed_providers", ()))
            external_enabled = getattr(self.config, "external_enabled", getattr(self.config, "commercial_enabled", False))
            legacy_open = getattr(self.config, "commercial_enabled", False) and not allowed
            if not external_enabled or (allowed and role.provider not in allowed) or (not allowed and not legacy_open):
                if not getattr(self.config, "external_enabled", False) and getattr(self.config, "commercial_enabled", False) is False:
                    raise RuntimeError(f"Commercial inference provider '{role.provider}' is disabled; set [commercial].enabled = true")
                raise RuntimeError(f"External inference provider '{role.provider}' is disabled or not allowlisted")
        key = (role.provider, role.server_profile if role.provider == "local" else self.config.base_url)
        if key in self._clients:
            return self._clients[key]
        if self.injected_client is not None:
            client = self.injected_client
        else:
            try:
                from cognityx_inference.client import CognityxInferenceClient
            except ModuleNotFoundError as exc:
                raise RuntimeError("Inference support is not installed. Install `cognityx-dataforge[inference]` or inject an inference client.") from exc
            local = role.provider == "local"
            client = CognityxInferenceClient(
                self.config.base_url,
                manager_url=self.config.manager_url,
                auto_start=bool(local and self.config.auto_start_local),
                startup_timeout_seconds=self.config.startup_timeout_seconds,
                backend=role.backend if local else None,
                profile=role.server_profile if local else None,
            )
        self._clients[key] = client
        self._preflight[key] = self._inspect_provider(client, role)
        return client

    def _inspect_provider(self, client: Any, role: GeneratorConfig) -> dict[str, Any]:
        if role.provider == "local":
            return {"provider": "local", "verification_source": "local-client-configuration", "capabilities": {}}
        snapshot: dict[str, Any] = {"provider": role.provider, "model": role.model, "verification_source": "provider-capability-endpoints"}
        for name, args in (("provider_status", ()), ("provider_capabilities", (role.provider, role.model))):
            method = getattr(client, name, None)
            if method is None:
                if self.injected_client is not None:
                    return {"provider": role.provider, "model": role.model, "verification_source": "injected-client"}
                raise RuntimeError(f"Inference client does not support required provider preflight: {name}")
            try:
                snapshot[name] = method(*args)
            except Exception as exc:
                raise RuntimeError(f"Provider preflight failed for '{role.provider}': {exc}") from exc
        capabilities = snapshot.get("provider_capabilities") or {}
        if capabilities.get("model_discovery", capabilities.get("models_supported", False)):
            method = getattr(client, "provider_models", None)
            if method is not None:
                snapshot["provider_models"] = method(role.provider)
        snapshot["parameter_policy"] = capabilities.get("parameters", {})
        return snapshot

    def preflight_for(self, role: GeneratorConfig) -> dict[str, Any]:
        self.client_for(role)
        return dict(self._preflight[(role.provider, role.server_profile if role.provider == "local" else self.config.base_url)])


class StructuredAdapter:
    def __init__(self, client_or_pool: Any, config: GeneratorConfig) -> None:
        self.pool = client_or_pool if isinstance(client_or_pool, InferenceClientPool) else None
        self.client = client_or_pool if self.pool is None else None
        self.config = config

    def _client(self) -> Any:
        return self.pool.client_for(self.config) if self.pool else self.client

    def _preflight(self) -> dict[str, Any]:
        return self.pool.preflight_for(self.config) if self.pool else {"provider": self.config.provider, "verification_source": "injected-client"}

    def ask(self, prompt: str, system: str) -> str:
        kwargs = {"model": self.config.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "provider": self.config.provider}
        if self.config.provider == "local":
            kwargs.update(backend=self.config.backend or "vllm", profile=self.config.profile or "bf16")
        if self.config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.config.max_output_tokens
        return _content(self._client().chat(**kwargs))

    def ask_budgeted(self, prompt: str, system: str, *, context_limit: int | None, role: str, prompt_version: str, evidence_ids: list[str], calls: list[dict[str, Any]]) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        kwargs = {"model": self.config.model, "messages": messages, "provider": self.config.provider}
        if self.config.provider == "local":
            kwargs.update(backend=self.config.backend or "vllm", profile=self.config.profile or "bf16")
        if self.config.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.config.max_output_tokens
        kwargs["request_metadata"] = {"model_role": role, "prompt_version": prompt_version, "evidence_ids": list(evidence_ids), "history_mode": "none"}
        policy = self._preflight()
        capabilities = policy.get("provider_capabilities", {})
        if capabilities.get("structured_output", capabilities.get("structured_outputs", False)):
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client().chat(**kwargs)
        except Exception as exc:
            detail = str(exc)
            if "context" in detail.lower() or "token" in detail.lower() or getattr(exc, "status", None) == 422:
                raise TokenBudgetError(detail) from exc
            raise
        if self.config.provider == "local" and context_limit is not None and self.config.max_output_tokens is not None:
            budget = (response.get("cognityx", {}) if isinstance(response, dict) else {}).get("token_budget")
            if budget is None:
                counter = getattr(self._client(), "count_input_tokens", None)
                if counter is not None:
                    counted = normalize_input_token_count(counter(model=self.config.model, messages=messages, backend=self.config.backend or "vllm", profile=self.config.profile or "bf16"))
                    if counted is not None and counted + self.config.max_output_tokens > context_limit:
                        raise TokenBudgetError("Model request exceeds context budget", input_tokens=counted, max_output_tokens=self.config.max_output_tokens, context_limit=context_limit)
        calls.append({**response_metadata(response, self.config, role), "profile_state": policy.get("provider_status", "configured"), "verification_source": policy.get("verification_source"), "capability_snapshot": capabilities, "parameter_policy": policy.get("parameter_policy", {}), "prompt_version": prompt_version, "evidence_ids": list(evidence_ids)})
        return _content(response)


class GeneratorAdapter:
    def __init__(self, client: Any, config: GeneratorConfig) -> None:
        self.adapter = StructuredAdapter(client, config)
        self.config = config

    def generate(self, prompt: str) -> dict[str, str]:
        data = json.loads(self.adapter.ask(prompt, "Return strict JSON with keys instruction and answer."))
        if not isinstance(data, dict) or not data.get("instruction") or not data.get("answer"):
            raise ValueError("Incomplete generator output")
        return {"instruction": str(data["instruction"]).strip(), "answer": str(data["answer"]).strip()}


def load_inference_client() -> Any:
    try:
        from cognityx_inference.client import CognityxInferenceClient
    except ModuleNotFoundError as exc:
        raise RuntimeError("Inference support is not installed. Install `cognityx-dataforge[inference]` or inject an inference client.") from exc
    return CognityxInferenceClient()
