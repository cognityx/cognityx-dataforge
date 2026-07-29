import pytest


def test_installed_inference_client_contract_is_constructible():
    try:
        from cognityx_inference.client import CognityxInferenceClient
    except ModuleNotFoundError:
        pytest.skip("inference extra is not installed")
    client = CognityxInferenceClient("http://127.0.0.1:9")
    for method in ("chat", "list_providers", "provider_status", "provider_models", "provider_capabilities", "test_provider", "server_status"):
        assert callable(getattr(client, method, None)), method
