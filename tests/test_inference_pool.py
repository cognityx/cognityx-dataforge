from types import SimpleNamespace

import pytest

from cognityx_dataforge.inference import GeneratorConfig, InferenceClientPool, StructuredAdapter


class Client:
    def __init__(self):
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "req-1",
            "choices": [{"message": {"content": '{"instruction":"i","answer":"a"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "cognityx": {"token_budget": {"effective_max_output_tokens": 8}, "extensions": {"rate_limits": {"remaining": "9"}}},
        }

    def provider_status(self):
        return [{"provider": "gemini", "configured": True}]

    def provider_capabilities(self, provider, model):
        return {"structured_output": True, "model_discovery": False}


def settings(enabled=False):
    return SimpleNamespace(base_url="http://gateway", manager_url="http://manager", auto_start_local=True, startup_timeout_seconds=5, commercial_enabled=enabled)


def test_injected_client_is_reused_and_calls_are_stateless():
    client = Client()
    pool = InferenceClientPool(config=settings(), injected_client=client)
    role = GeneratorConfig(model="m", provider="local", backend="vllm", profile="int4", server_profile="gpu")
    adapter = StructuredAdapter(pool, role)
    calls = []
    adapter.ask_budgeted("one", "system", context_limit=None, role="qa", prompt_version="1", evidence_ids=["e1"], calls=calls)
    adapter.ask_budgeted("two", "system", context_limit=None, role="qa", prompt_version="1", evidence_ids=["e2"], calls=calls)
    assert len(client.calls) == 2
    assert all(len(call["messages"]) == 2 for call in client.calls)
    assert calls[0]["request_id"] == "req-1"
    assert calls[0]["rate_limit_diagnostics"] == {"remaining": "9"}


def test_commercial_guard_happens_before_dispatch():
    client = Client()
    pool = InferenceClientPool(config=settings(), injected_client=client)
    with pytest.raises(RuntimeError, match="Commercial inference provider 'groq' is disabled"):
        StructuredAdapter(pool, GeneratorConfig(model="m", provider="groq")).ask("x", "s")
    assert not client.calls


def test_commercial_role_can_route_without_local_fields():
    client = Client()
    pool = InferenceClientPool(config=settings(enabled=True), injected_client=client)
    StructuredAdapter(pool, GeneratorConfig(model="m", provider="openai")).ask("x", "s")
    assert client.calls[0]["provider"] == "openai"
    assert "backend" not in client.calls[0]


def test_arbitrary_external_provider_is_preflighted_and_uses_supported_parameters():
    client = Client()
    pool = InferenceClientPool(config=SimpleNamespace(base_url="http://gateway", manager_url=None, auto_start_local=False, startup_timeout_seconds=5, external_enabled=True, allowed_providers=("gemini",), commercial_enabled=False), injected_client=client)
    calls = []
    StructuredAdapter(pool, GeneratorConfig(model="gemini-pro", provider="gemini", max_output_tokens=12)).ask_budgeted("x", "s", context_limit=None, role="judge", prompt_version="2", evidence_ids=[], calls=calls)
    assert client.calls[0]["provider"] == "gemini"
    assert client.calls[0]["max_output_tokens"] == 12
    assert "max_tokens" not in client.calls[0]
    assert client.calls[0]["response_format"] == {"type": "json_object"}
