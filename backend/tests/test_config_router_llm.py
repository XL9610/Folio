"""Tests for LLM settings persistence behavior (environment-only)."""

from __future__ import annotations

import asyncio

import pytest

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


@pytest.mark.parametrize(
    "arg0,arg1,arg2,arg3,arg4,arg5",
    [
        ("openai", "https://api.openai.com/v1", "google", "google", "", "gemini-test"),
        ("custom", "https://old.example.com/v1", "custom", "openai-compatible", "https://new.example.com/v1", "my-model"),
    ],
    ids=['save_llm_config_clears_key_on_provider_switch', 'save_llm_config_clears_key_on_endpoint_change'],
)
def test_save_llm_config_clears_key_on(monkeypatch, arg0, arg1, arg2, arg3, arg4, arg5):
    saved = {}

    def fake_set_env(key, value):
        saved[key] = value

    monkeypatch.setattr(config_router, "set_env", fake_set_env)
    monkeypatch.setattr(config_router, "_reset_cached_llm_services", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", arg0)
    monkeypatch.setenv("LLM_BASE_URL", arg1)
    monkeypatch.setenv("LLM_API_FORMAT", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "old-key")

    asyncio.run(config_router.save_llm_config(LLMConfigRequest(
        provider=arg2,
        api_format=arg3,
        api_key="",
        base_url=arg4,
        model=arg5,
    )))

    # Switching provider with a blank key must not reuse the old cloud key.
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
