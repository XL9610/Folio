"""Tests for LLM settings persistence behavior (environment-only)."""

from __future__ import annotations

import asyncio

from app.routers import config_router
from app.routers.config_router import LLMConfigRequest


def test_llm_config_reads_env(monkeypatch):
    from app.services.llm import LLMConfig

    monkeypatch.setenv("LLM_REMOTE_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_FORMAT", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://anthropic.test/v1")
    monkeypatch.setenv("LLM_MODEL", "claude-test")

    cfg = LLMConfig()
    assert cfg.is_configured is True
    assert cfg.provider == "anthropic"
    assert cfg.api_format == "anthropic"
    assert cfg.api_key == "env-key"
    assert cfg.model == "claude-test"


def test_llm_config_unconfigured_without_key(monkeypatch):
    from app.services.llm import LLMConfig

    monkeypatch.setenv("LLM_REMOTE_ENABLED", "true")
    monkeypatch.setenv("LLM_API_KEY", "")
    assert LLMConfig().is_configured is False


def test_save_llm_config_keeps_existing_key_and_resets_caches(monkeypatch):
    saved = {}
    reset = {"called": False}

    def fake_set_env(key, value):
        saved[key] = value

    monkeypatch.setattr(config_router, "set_env", fake_set_env)
    monkeypatch.setattr(config_router, "_reset_cached_llm_services", lambda: reset.update(called=True))
    monkeypatch.setenv("LLM_PROVIDER", "google")  # same endpoint → key kept
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    monkeypatch.setenv("LLM_API_FORMAT", "google")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    result = asyncio.run(config_router.save_llm_config(LLMConfigRequest(
        provider="google",
        api_format="google",
        api_key="",
        base_url="",
        model="gemini-test",
    )))

    assert result["configured"] is True
    assert saved["LLM_API_KEY"] == "old-key"
    assert saved["LLM_API_FORMAT"] == "google"
    assert saved["LLM_BASE_URL"] == "https://generativelanguage.googleapis.com/v1beta"
    assert saved["LLM_MODEL"] == "gemini-test"
    assert saved["LLM_REMOTE_ENABLED"] == "true"
    assert reset["called"] is True


def test_save_llm_config_clears_key_on_provider_switch(monkeypatch):
    saved = {}

    def fake_set_env(key, value):
        saved[key] = value

    monkeypatch.setattr(config_router, "set_env", fake_set_env)
    monkeypatch.setattr(config_router, "_reset_cached_llm_services", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_FORMAT", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    asyncio.run(config_router.save_llm_config(LLMConfigRequest(
        provider="google",
        api_format="google",
        api_key="",
        base_url="",
        model="gemini-test",
    )))

    # Switching provider with a blank key must not reuse the old cloud key.
    assert saved["LLM_API_KEY"] == ""


def test_save_llm_config_clears_key_on_endpoint_change(monkeypatch):
    saved = {}

    def fake_set_env(key, value):
        saved[key] = value

    monkeypatch.setattr(config_router, "set_env", fake_set_env)
    monkeypatch.setattr(config_router, "_reset_cached_llm_services", lambda: None)
    # Same provider, but a different base URL → must not reuse the old key.
    monkeypatch.setenv("LLM_PROVIDER", "custom")
    monkeypatch.setenv("LLM_BASE_URL", "https://old.example.com/v1")
    monkeypatch.setenv("LLM_API_FORMAT", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    asyncio.run(config_router.save_llm_config(LLMConfigRequest(
        provider="custom",
        api_format="openai-compatible",
        api_key="",
        base_url="https://new.example.com/v1",
        model="my-model",
    )))

    assert saved["LLM_API_KEY"] == ""


def test_save_llm_config_local_needs_no_key(monkeypatch):
    saved = {}

    def fake_set_env(key, value):
        saved[key] = value

    monkeypatch.setattr(config_router, "set_env", fake_set_env)
    monkeypatch.setattr(config_router, "_reset_cached_llm_services", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    result = asyncio.run(config_router.save_llm_config(LLMConfigRequest(
        provider="local",
        api_format="openai-compatible",
        api_key="",
        base_url="",
        model="llama3",
    )))

    assert saved["LLM_API_KEY"] == ""
    assert result["configured"] is True  # local needs no key
