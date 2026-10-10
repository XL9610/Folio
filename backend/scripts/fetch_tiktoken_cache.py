"""Prefetch the tiktoken BPE cache into backend/packaging/tiktoken_cache.

Run during packaging (like fetch_embedding_model.py) so the frozen bundle is
fully offline: the PDF chunker uses cl100k_base, which tiktoken otherwise
downloads from Azure on first use.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / "packaging" / "tiktoken_cache"


def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_CACHE_DIR)

    import tiktoken

    for name in ("cl100k_base",):
        try:
            tiktoken.get_encoding(name)
            print(f"cached {name} -> {_CACHE_DIR}")
        except Exception as exc:  # noqa: BLE001 — report and fail the build
            print(f"failed to cache {name}: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
