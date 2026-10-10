"""Tests for native provider LLM clients."""

from __future__ import annotations

from types import SimpleNamespace

from app.services.llm.client import AnthropicNativeClient, GoogleNativeClient, create_llm_client


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class FakeHTTPClient:
    response_data = {}
    last_request = None

    def __init__(self, timeout=None):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, **kwargs):
        FakeHTTPClient.last_request = {"url": url, **kwargs}
        return FakeResponse(FakeHTTPClient.response_data)


def cfg(provider, api_format, base_url, model):
    return SimpleNamespace(
        provider=provider,
        api_format=api_format,
        api_key="test-key",
        base_url=base_url,
        model=model,
        max_tokens=123,
        temperature=0.2,
        is_configured=True,
    )


def test_anthropic_native_client_parses_text(monkeypatch):
    monkeypatch.setattr("app.services.llm.client.httpx.Client", FakeHTTPClient)
    FakeHTTPClient.response_data = {"content": [{"type": "text", "text": "hello"}]}
    client = AnthropicNativeClient(cfg("anthropic", "anthropic", "https://anthropic.test/v1", "claude-test"))

    text = client.complete_chat([{"role": "user", "content": "Hi"}])

    assert text == "hello"
    assert FakeHTTPClient.last_request["url"] == "https://anthropic.test/v1/messages"
    assert FakeHTTPClient.last_request["headers"]["x-api-key"] == "test-key"


def test_google_native_client_parses_text(monkeypatch):
    monkeypatch.setattr("app.services.llm.client.httpx.Client", FakeHTTPClient)
    FakeHTTPClient.response_data = {
        "candidates": [{"content": {"parts": [{"text": "gemini says hi"}]}}]
    }
    client = GoogleNativeClient(cfg("google", "google", "https://google.test/v1beta", "gemini-test"))

    text = client.complete_chat([{"role": "user", "content": "Hi"}])

    assert text == "gemini says hi"
    assert FakeHTTPClient.last_request["url"] == "https://google.test/v1beta/models/gemini-test:generateContent"
    # Key must travel in the header, never in the query string (which would leak
    # into error messages, logs, and the connection-test response).
    assert FakeHTTPClient.last_request["headers"]["x-goog-api-key"] == "test-key"
    assert "key" not in FakeHTTPClient.last_request.get("params", {})


def test_factory_selects_native_clients():
    assert create_llm_client(cfg("anthropic", "anthropic", "", "claude")).__class__ is AnthropicNativeClient
    assert create_llm_client(cfg("google", "google", "", "gemini")).__class__ is GoogleNativeClient
