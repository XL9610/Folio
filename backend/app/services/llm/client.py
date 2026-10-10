"""Provider-aware LLM clients.

OpenAI-compatible providers use the OpenAI SDK. Anthropic and Google use their
native HTTP APIs through httpx so the Settings UI can configure them directly
without pretending they are OpenAI-compatible endpoints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from . import LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMJSONResponse:
    """Result from calling the LLM with a JSON-mode request."""

    data: dict
    raw_text: str
    model_used: str
    finish_reason: str


def _extract_json_object(raw: str) -> dict:
    """Parse a JSON object from provider output, tolerating fences/preamble."""
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("LLM did not return a valid JSON object.")


class OpenAICompatibleClient:
    """Thin wrapper around the OpenAI client for OpenAI-compatible APIs."""

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig()
        api_key = self._config.api_key
        if (self._config.provider or "").strip().lower() == "local":
            api_key = "local"  # local servers ignore the key; never forward a cloud key
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url
        self._client = OpenAI(**kwargs)

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self._config.is_configured:
            raise ValueError("LLM_API_KEY is not set. Cannot invoke LLM.")

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature if temperature is not None else self._config.temperature,
            )
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            cause_str = f"{type(cause).__name__}: {cause}" if cause else "(no underlying cause)"
            logger.error("LLM API call failed: %s | underlying: %s", exc, cause_str)
            msg = str(exc)
            if "Connection error" in msg or "connection" in msg.lower():
                msg = (
                    f"Connection error — cannot reach {self._config.base_url}. "
                    f"Underlying: {cause_str}. "
                    "Check that the LLM provider is online and your network allows outbound HTTPS."
                )
            raise RuntimeError(f"LLM API call failed: {msg}") from exc

        choice = response.choices[0]
        return choice.message.content or ""

    def complete_json(self, system_prompt: str, user_prompt: str) -> LLMJSONResponse:
        if not self._config.is_configured:
            raise ValueError("LLM_API_KEY is not set. Cannot invoke LLM.")

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("LLM API call failed: %s", exc)
            msg = str(exc)
            # Surface actionable diagnostics for connection errors
            if "Connection error" in msg or "connection" in msg.lower():
                msg = (
                    f"Connection error — cannot reach {self._config.base_url}. "
                    "Check that the LLM provider is online and your network allows outbound HTTPS. "
                    f"Detail: {exc}"
                )
            raise RuntimeError(f"LLM API call failed: {msg}") from exc

        choice = response.choices[0]
        raw = choice.message.content or "{}"
        finish = choice.finish_reason or "unknown"
        try:
            data = _extract_json_object(raw)
        except ValueError as exc:
            logger.error("LLM returned invalid JSON (model=%s, finish=%s)", self._config.model, finish)
            raise ValueError(f"LLM did not return valid JSON: {exc}") from exc

        return LLMJSONResponse(data=data, raw_text=raw, model_used=self._config.model, finish_reason=finish)


class AnthropicNativeClient:
    """Anthropic Messages API client."""

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig()
        self._base_url = (self._config.base_url or "https://api.anthropic.com/v1").rstrip("/")

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self._config.is_configured:
            raise ValueError("LLM_API_KEY is not set. Cannot invoke LLM.")

        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                api_messages.append({"role": role, "content": content})
            else:
                api_messages.append({"role": "user", "content": content})

        body: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "messages": api_messages or [{"role": "user", "content": ""}],
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self._config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(f"{self._base_url}/messages", headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("Anthropic API call failed: %s", exc)
            raise RuntimeError(f"Anthropic API call failed: {exc}") from exc

        parts = data.get("content", [])
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in parts
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> LLMJSONResponse:
        raw = self.complete_chat(
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\nReturn only a valid JSON object."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return LLMJSONResponse(
            data=_extract_json_object(raw),
            raw_text=raw,
            model_used=self._config.model,
            finish_reason="unknown",
        )


class GoogleNativeClient:
    """Google Gemini generateContent API client."""

    def __init__(self, config: LLMConfig | None = None):
        self._config = config or LLMConfig()
        self._base_url = (self._config.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self._config.is_configured:
            raise ValueError("LLM_API_KEY is not set. Cannot invoke LLM.")

        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
                continue
            contents.append({
                "role": "model" if role == "assistant" else "user",
                "parts": [{"text": content}],
            })

        body: dict[str, Any] = {
            "contents": contents or [{"role": "user", "parts": [{"text": ""}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
            },
        }
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        url = f"{self._base_url}/models/{self._config.model}:generateContent"
        # Send the key in the x-goog-api-key header (not a query param) so it
        # never appears in the request URL, which would otherwise leak into
        # error messages, backend.log, and the connection-test response.
        headers = {"x-goog-api-key": self._config.api_key}
        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.error("Google Gemini API call failed (http_status=%s)", status)
            detail = f" (HTTP {status})" if status else ""
            raise RuntimeError(f"Google Gemini API call failed{detail}") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in parts
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> LLMJSONResponse:
        raw = self.complete_chat(
            messages=[
                {"role": "system", "content": f"{system_prompt}\n\nReturn only a valid JSON object."},
                {"role": "user", "content": user_prompt},
            ],
        )
        return LLMJSONResponse(
            data=_extract_json_object(raw),
            raw_text=raw,
            model_used=self._config.model,
            finish_reason="unknown",
        )


def create_llm_client(config: LLMConfig | None = None):
    """Create the right client for the configured provider/API format."""
    cfg = config or LLMConfig()
    fmt = (cfg.api_format or "openai-compatible").strip().lower()
    provider = (cfg.provider or "").strip().lower()
    if fmt in {"anthropic", "anthropic-native"} or provider == "anthropic":
        return AnthropicNativeClient(cfg)
    if fmt in {"google", "google-native", "gemini"} or provider == "google":
        return GoogleNativeClient(cfg)
    return OpenAICompatibleClient(cfg)
