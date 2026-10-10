"""PyInstaller entry point — starts the FastAPI backend.

Kept as a plain module (imports `app` by string-free reference) so PyInstaller
can statically trace the dependency graph. Runs uvicorn directly against the
FastAPI app object (no `--reload`, no import-string indirection).

The windowed build (console=False) has no console, so stdout/stderr are unusable
and import-time logging would hit a None stream. Redirect them to a log file in
the per-user data dir before anything else runs.
"""

import logging
import os
import sys

from app.core.env_file import user_data_dir

# Redirect stdout/stderr and add a root-logger file handler before importing the
# app: in windowed mode sys.stdout/sys.stderr are None, and any logging before
# this redirect would either vanish or raise.
_LOG_DIR = user_data_dir() / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = open(_LOG_DIR / "backend.log", "a", encoding="utf-8", buffering=1)
sys.stdout = _LOG_FILE
sys.stderr = _LOG_FILE

_LOGGING_HANDLER = logging.StreamHandler(_LOG_FILE)
_LOGGING_HANDLER.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
)
logging.getLogger().addHandler(_LOGGING_HANDLER)
logging.getLogger().setLevel(logging.INFO)

# tiktoken downloads its BPE encodings on first use; the frozen bundle ships a
# pre-populated cache so a fresh offline install never reaches the network.
if getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(_bundle_dir, "tiktoken_cache"))

import uvicorn  # noqa: E402
from app.main import app  # noqa: E402


def main() -> None:
    # Honor PORT so the build script's smoke test can avoid clashing with a
    # dev backend already on 8000. Defaults to the standard port.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
