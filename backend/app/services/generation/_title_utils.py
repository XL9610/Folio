"""Shared title / filename cleaning helpers for generated artifacts."""

from __future__ import annotations

import re

_PIRACY_PATTERNS = [
    r"_?z[_-]?library[_-]?sk[_,.\s]*",
    r"_?1lib[_-]?sk[_,.\s]*",
    r"_?libgen[_-]?[a-z]*[_,.\s]*",
    r"_?anna[_-]?archive[_,.\s]*",
    r"\(z[_-]?lib[^)]*\)",
    r"\[z[_-]?lib[^\]]*\]",
]

_EXTENSION_RE = re.compile(r"\.(pdf|epub|mobi|djvu|txt|docx?)$", re.IGNORECASE)


def clean_filename(filename: str | None) -> str:
    """Turn a messy PDF filename into a readable book title.

    Examples:
        "AI_Engineering_Building_Applications_z_library_sk,_1lib_sk,.pdf"
          -> "AI Engineering Building Applications"
        "A History of Western Music Tenth.pdf"
          -> "A History of Western Music Tenth"
    """
    if not filename:
        return "Untitled document"

    s = filename.strip()
    s = _EXTENSION_RE.sub("", s)

    for pat in _PIRACY_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)

    s = s.replace("_", " ")
    s = re.sub(r"[,;\-\s]+$", "", s)
    s = re.sub(r"^[,;\-\s]+", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    return s or "Untitled document"