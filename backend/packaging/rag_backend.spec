# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the RAG Study Companion backend.

Produces an onedir bundle (folder) so electron-builder can drop the whole thing
into extraResources in the next step. `sqlite_vec` is the one package that
ships a native extension (vec0.dll) the app loads via `sqlite_vec.load()`; the
rest of the `collect_all` calls handle data files + dynamic imports that
PyInstaller's static analysis would otherwise miss (transformers/tokenizers/
tiktoken model configs, sentence-transformers, torch's native libs).

Build from the repo root (CPU-torch venv recommended):

    pyinstaller backend/packaging/rag_backend.spec
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

import os

# Paths are resolved relative to this spec file (backend/packaging/), so anchor
# the entry point and package root to it rather than to the build's CWD.
_BACKEND_DIR = os.path.abspath(os.path.join(SPECPATH, ".."))

datas = []
binaries = []
hiddenimports = []

# sqlite-vec must keep vec0.dll next to its __init__.py.
# torch/transformers/tokenizers/sentence-transformers carry data + native libs.
for pkg in (
    "sqlite_vec",
    "torch",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "tiktoken",
    "tiktoken_ext",
    "ebooklib",
    "pptx",
    "docx",
    "kreuzberg",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# tiktoken discovers its encoding plugins at runtime via
# pkgutil.iter_modules(tiktoken_ext.__path__), so openai_public.py must exist as
# a real file on disk (not only inside the PYZ archive). Bundle the .py source
# as data so the discovery finds it; the hidden import above makes it importable.
datas += collect_data_files("tiktoken_ext", include_py_files=True)

# uvicorn[standard] pulls optional loop/protocol backends via dynamic import.
hiddenimports += collect_submodules("uvicorn")

# Bundle the embedding model weights so the frozen app works fully offline.
# backend/scripts/fetch_embedding_model.py downloads them into this directory.
_EMBEDDING_MODEL_DIR = os.path.join(_BACKEND_DIR, "packaging", "embedding_model")
if os.path.isdir(_EMBEDDING_MODEL_DIR):
    datas.append((_EMBEDDING_MODEL_DIR, "embedding_model"))

# Bundle the tiktoken BPE cache so the frozen app chunks PDFs fully offline.
# backend/scripts/fetch_tiktoken_cache.py downloads it into this directory.
# Fail the build if it is missing rather than silently shipping a bundle that
# would try to reach the network on first PDF.
_TIKTOKEN_CACHE_DIR = os.path.join(_BACKEND_DIR, "packaging", "tiktoken_cache")
if not os.path.isdir(_TIKTOKEN_CACHE_DIR):
    raise SystemExit(
        "tiktoken cache missing: run backend/scripts/fetch_tiktoken_cache.py first"
    )
datas.append((_TIKTOKEN_CACHE_DIR, "tiktoken_cache"))

a = Analysis(
    [os.path.join(_BACKEND_DIR, "launcher.py")],
    pathex=[_BACKEND_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # No torch module excludes. The high-value, zero-risk slimming is the
    # _keep_entry() datas/binaries filter above (C++ headers + import libraries).
    # Excluding torch submodules isn't worth it: torch.distributed / torch.testing
    # are imported eagerly by `import torch` (would break the frozen app), and
    # torch._inductor / _dynamo are listed by hook-torch so excluding them floods
    # the build log with "hidden import not found" noise for ~17MB of savings.
    excludes=["pytest", "playwright"],
    noarchive=False,
)

# Slim the bundle AFTER analysis: hook-torch adds .lib import libraries and
# include/ C++ headers on top of what collect_all gathered, and those only land
# in a.datas/a.binaries once analysis runs. TOC entries are
# (dest_name, src_path, typecode). .lib/.a are linker-time only, and the C++
# headers are only needed to compile extensions — never at runtime for CPU
# inference.
def _is_dead_weight(f1, f2):
    f1 = f1.lower().replace("\\", "/")
    f2 = f2.lower().replace("\\", "/")
    if f1.endswith((".lib", ".a", ".pdb")) or f2.endswith((".lib", ".a", ".pdb")):
        return True
    return "torch/include/" in f1 or "torch/include/" in f2


# Mutate in place (TOC objects) to preserve their type flag — reassigning a
# plain list here would silently drop them from COLLECT.
a.datas[:] = [e for e in a.datas if not _is_dead_weight(e[0], e[1])]
a.binaries[:] = [e for e in a.binaries if not _is_dead_weight(e[0], e[1])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rag-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True,
    upx=False,
    name="rag-backend",
)
