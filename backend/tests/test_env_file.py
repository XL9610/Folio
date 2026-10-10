"""Tests for .env path resolution (dev vs frozen) and set_env."""

import importlib
import os
import sys
from pathlib import Path

import app.core.env_file as env_file


def test_user_data_dir_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:/Users/u/AppData/Roaming")
    assert env_file.user_data_dir() == Path("C:/Users/u/AppData/Roaming") / "Folio"


def test_set_env_writes_file_and_creates_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(env_file, "ENV_PATH", tmp_path / "nested" / "deep" / ".env")
    env_file.set_env("TEST_ENV_KEY", "abc123")
    target = tmp_path / "nested" / "deep" / ".env"
    assert target.exists()
    text = target.read_text(encoding="utf-8")
    assert "TEST_ENV_KEY" in text and "abc123" in text
    assert os.environ["TEST_ENV_KEY"] == "abc123"


def test_env_path_frozen_uses_user_dir(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", "C:/Users/u/AppData/Roaming")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    reloaded = importlib.reload(env_file)
    try:
        assert reloaded.ENV_PATH == Path("C:/Users/u/AppData/Roaming") / "Folio" / ".env"
    finally:
        # Reload back to dev mode so later tests see backend/.env.
        monkeypatch.setattr(sys, "frozen", False, raising=False)
        importlib.reload(env_file)
