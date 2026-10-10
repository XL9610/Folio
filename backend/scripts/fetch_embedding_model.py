"""Download the embedding model weights into the packaging tree.

Run once before building the desktop app (scripts/build-desktop.ps1 calls this).
Snapshots BAAI/bge-small-en-v1.5 into backend/packaging/embedding_model/ so the
PyInstaller spec can bundle it and the frozen app loads it fully offline.

Supports the HF_ENDPOINT env var for a mirror (e.g. https://hf-mirror.com) —
useful where huggingface.co is unreachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
OUT_DIR = Path(__file__).resolve().parent.parent / "packaging" / "embedding_model"


def main() -> int:
    from huggingface_hub import snapshot_download

    if (OUT_DIR / "config.json").exists():
        print(f"embedding model already present at {OUT_DIR}")
        return 0

    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    print(f"Downloading {MODEL_ID} -> {OUT_DIR} (HF_ENDPOINT={endpoint})")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=OUT_DIR,
        # Skip the legacy/alternate weight formats — sentence-transformers loads
        # the .safetensors copy, so bundling the .bin and .onnx duplicates would
        # ~3x the model size (~127MB each).
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.ot", "*.onnx"],
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
