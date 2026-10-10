"""LLM configuration — environment-only (no DB persistence).

Remote LLM is disabled by default (local-first).
Set LLM_REMOTE_ENABLED=true or configure via Settings UI (which writes .env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from ...core.env_file import ENV_PATH

# Auto-load .env (dev: backend/.env; frozen: %APPDATA%/Folio/.env)
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration, read from environment variables only.

    Priority: environment variables (or .env) > defaults.
    """

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", ""))
    api_format: str = field(default_factory=lambda: os.getenv("LLM_API_FORMAT", "openai-compatible"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))

    @property
    def is_configured(self) -> bool:
        """Return True if the LLM is enabled and has the credentials it needs.

        Local providers (Ollama/LM Studio) need no API key; remote providers do.
        """
        enabled = os.getenv("LLM_REMOTE_ENABLED", "false").lower() == "true"
        if not enabled:
            return False
        if (self.provider or "").strip().lower() == "local":
            return True
        return bool(self.api_key)
