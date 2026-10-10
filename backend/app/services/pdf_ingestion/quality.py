"""Ingestion quality gate — flags suspicious chunk outputs.

Run after chunking to detect when body-window detection may have
missed front/back matter, preventing silent failures.
"""

from __future__ import annotations

import re
from .models import Chunk

# Patterns that indicate non-body content in the first few chunks
_FRONT_MATTER_SIGNALS: list[re.Pattern] = [
    re.compile(r"^(?:front\s*)?cover$", re.IGNORECASE),
    re.compile(r"^title\s*page$", re.IGNORECASE),
    re.compile(r"^copyright$", re.IGNORECASE),
    re.compile(r"^contents$", re.IGNORECASE),
    re.compile(r"^table\s*of\s*contents$", re.IGNORECASE),
    re.compile(r"^editorial", re.IGNORECASE),
    re.compile(r"^advisory\s+board$", re.IGNORECASE),
    re.compile(r"^maps?$", re.IGNORECASE),
    re.compile(r"^guide\s+to\s+", re.IGNORECASE),
    re.compile(r"^preface", re.IGNORECASE),
    re.compile(r"^acknowledg?ments?$", re.IGNORECASE),
]

_BACK_MATTER_SIGNALS: list[re.Pattern] = [
    re.compile(r"^bibliography$", re.IGNORECASE),
    re.compile(r"^references?$", re.IGNORECASE),
    re.compile(r"^works?\s*cited$", re.IGNORECASE),
    re.compile(r"^further\s*reading$", re.IGNORECASE),
    re.compile(r"^glossary$", re.IGNORECASE),
    re.compile(r"^index", re.IGNORECASE),
    re.compile(r"^credits$", re.IGNORECASE),
    re.compile(r"^colophon$", re.IGNORECASE),
    re.compile(r"^about\s*the\s*(?:author|authors)", re.IGNORECASE),
    re.compile(r"^epilogue$", re.IGNORECASE),
]


def _title_matches(patterns: list[re.Pattern], title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    return any(p.match(t) for p in patterns)


def assess_chunk_quality(chunks: list[Chunk], page_count: int = 0) -> list[str]:
    """Return warnings about suspicious chunk quality.

    Specifically checks if the first/last chunks look like front/back matter,
    which would indicate body-window detection failure.
    """
    warnings: list[str] = []

    if not chunks:
        warnings.append("No chunks produced — PDF may be empty or scanned image.")
        return warnings

    # Check first chunks for front-matter signals
    front_check_count = min(10, len(chunks))
    front_hits = 0
    for c in chunks[:front_check_count]:
        if _title_matches(_FRONT_MATTER_SIGNALS, c.title):
            front_hits += 1

    if front_hits > front_check_count * 0.4:
        warnings.append(
            f"Quality gate: {front_hits}/{front_check_count} first chunks appear to be "
            f"front matter (title page, copyright, contents, etc.). "
            f"Body-window detection may have failed — review body boundary."
        )

    # Check first chunk specifically
    if chunks and _title_matches(_FRONT_MATTER_SIGNALS, chunks[0].title):
        warnings.append(
            f"Quality gate: first chunk title is \"{chunks[0].title}\", "
            f"which looks like front matter. Body-window may not have started correctly."
        )

    # Check last chunks for back-matter signals
    back_check_count = min(10, len(chunks))
    back_hits = 0
    for c in chunks[-back_check_count:]:
        if _title_matches(_BACK_MATTER_SIGNALS, c.title):
            back_hits += 1

    if back_hits > back_check_count * 0.4:
        warnings.append(
            f"Quality gate: {back_hits}/{back_check_count} last chunks appear to be "
            f"back matter (bibliography, index, glossary, credits, etc.). "
            f"Body-window detection may not have captured back-matter boundary."
        )

    # Check if total body chunks seems too few relative to pages
    if page_count > 0 and len(chunks) < max(5, page_count * 0.5):
        warnings.append(
            f"Quality gate: only {len(chunks)} chunks for {page_count} pages. "
            f"Chunk count seems low — PDF may have limited extractable text."
        )

    return warnings
