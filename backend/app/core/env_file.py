"""Canonical .env path and write helper — the single source of config truth.

The Settings UI (config_router) and the one-time DB→env migration both write
here. Both config loaders (core/config.py and services/llm/__init__.py) anchor
to the same path. When running as a PyInstaller bundle, this path lives in the
per-user OS data directory (e.g. ``%APPDATA%/Folio/.env``) instead of next to the
(possibly read-only, replaced-on-update) executable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import set_key as _dotenv_set_key


def user_data_dir() -> Path:
    """Per-user writable data directory (used when frozen)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "Folio"


if getattr(sys, "frozen", False):
    # PyInstaller bundle: settings persist in the per-user data dir.
    ENV_PATH = user_data_dir() / ".env"
else:
    # Dev: backend/.env
    ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def set_env(key: str, value: str) -> None:
    """Persist key=value to ENV_PATH and update the live process env.

    dotenv.set_key preserves unrelated keys already in the file and creates the
    file if it is missing. The live os.environ update is what makes the value
    visible to same-process readers (AppConfig/LLMConfig read os.getenv).
    """
    os.environ[key] = value
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    _dotenv_set_key(ENV_PATH, key, value)
