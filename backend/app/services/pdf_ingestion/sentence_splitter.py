"""Sentence splitter — pure functions for splitting text into sentences.

No classes.  No external NLP dependencies (no nltk, no spaCy).
"""

from __future__ import annotations

import re

# Unicode private-use marker character for abbreviation-dot replacement.
# We use the Unicode REPLACEMENT CHARACTER (U+FFFD) which is extremely
# unlikely to appear in real PDF text, so a simple string replace is safe.
_ABBREV_PLACEHOLDER: str = "�"

# ---------------------------------------------------------------------------
# Protected abbreviations (case-insensitive matching)
# ---------------------------------------------------------------------------
_ABBREVIATIONS: tuple[str, ...] = (
    "Dr", "Mr", "Mrs", "Ms", "Prof",
    "e.g", "i.e",
    "etc",
    "vs",
    "Fig", "Ch", "Eq", "Sec", "Vol", "No",
    "approx",
    "et al",
    "U.S", "U.K", "Ph.D", "M.A", "B.A",
    "Inc", "Ltd", "Corp",
)

# Compiled pattern that matches any known abbreviation followed by a dot
# (e.g. "Dr.", "e.g.", "U.S.").
_ABBREV_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.(?!\w)",
    re.IGNORECASE,
)

# Matches sentence-ending punctuation: . ! ? followed by optional closing
# quote / paren / bracket, then whitespace, then a capital letter or digit
# or opening quote as the start of the next sentence.
_SENTENCE_BOUNDARY: re.Pattern[str] = re.compile(
    r'(?<=[.!?])(?:"|\'|\)|\]|})?\s+(?=[A-Z0-9"\'\(\[\{])',
)

# Matches decimal numbers (e.g. 3.14, 0.5) so we don't split inside them.
_DECIMAL_PATTERN: re.Pattern[str] = re.compile(
    r"\b\d+\.\d+\b",
)

# Separator heuristic: semicolons and colons between complete clauses.
# Capture the preceding word so we can verify length >= 3 in code
# (avoids variable-width lookbehind which Python's re does not support).
_CLAUSE_SEPARATOR: re.Pattern[str] = re.compile(
    r"(\w{3,})([;:])\s+(?=\w)",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _protect_abbreviations(text: str) -> str:
    """Replace abbreviation dots with a placeholder character so they are not
    treated as sentence boundaries."""
    def _replacer(m: re.Match[str]) -> str:
        return m.group(0)[:-1] + _ABBREV_PLACEHOLDER

    return _ABBREV_PATTERN.sub(_replacer, text)


def _restore_abbreviations(text: str) -> str:
    """Replace all abbreviation placeholder characters back to dots."""
    return text.replace(_ABBREV_PLACEHOLDER, ".")


def _protect_decimals(text: str) -> tuple[str, dict[str, str]]:
    """Replace decimal dots with placeholders. Returns protected text and
    a mapping from placeholder strings to original decimal strings."""
    mapping: dict[str, str] = {}
    counter = 0

    def _replacer(m: re.Match[str]) -> str:
        nonlocal counter
        key = f"￾DEC{counter}￾"
        counter += 1
        mapping[key] = m.group(0)
        return key

    protected = _DECIMAL_PATTERN.sub(_replacer, text)
    return protected, mapping


def _restore_decimals(text: str, mapping: dict[str, str]) -> str:
    """Restore decimal placeholders to original text."""
    result = text
    for key, original in mapping.items():
        result = result.replace(key, original)
    return result


def _protect_initials(text: str) -> tuple[str, dict[str, str]]:
    """Protect initial patterns like 'J. R. R.' by replacing the dots
    between single capital letters with placeholders.

    Pattern: single uppercase letter followed by a dot, spaced between
    other such initials (e.g., 'J. R. R. Tolkien').
    """
    _INITIAL_PATTERN = re.compile(
        r"\b([A-Z])\.(?:\s+([A-Z])\.)+(?=\s+[A-Z][a-z])",
    )
    mapping: dict[str, str] = {}
    counter = 0

    def _replacer(m: re.Match[str]) -> str:
        nonlocal counter
        full = m.group(0)
        key = f"￾INI{counter}￾"
        counter += 1
        mapping[key] = full
        return key

    protected = _INITIAL_PATTERN.sub(_replacer, text)
    return protected, mapping


def _restore_initials(text: str, mapping: dict[str, str]) -> str:
    """Restore initial placeholders."""
    result = text
    for key, original in mapping.items():
        result = result.replace(key, original)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using regex rules.

    Rules (in order):
    1. Protect known abbreviations: Dr., Mr., Mrs., Ms., Prof., e.g., i.e.,
       etc., vs., Fig., Ch., Eq., Sec., Vol., No., approx., et al.,
       U.S., U.K., Ph.D., M.A., B.A., Inc., Ltd., Corp.
    2. Protect decimal numbers: 3.14, 0.5
    3. Protect initials: J. R. R. Tolkien, H. G. Wells
    4. Split at sentence-ending punctuation (.!?) followed by:
       - optional closing quote/paren (\", ', ), ])
       - whitespace (space, newline)
       - next sentence starts with capital letter or digit or opening quote
    5. Handle multi-sentence per line: split on '; ' or ': ' when between
       complete clauses (heuristic: preceding word length > 2)
    6. Return list of non-empty sentence strings

    Returns an empty list for empty input.
    """
    if not text or not text.strip():
        return []

    # Phase 1: protect known patterns
    protected = _protect_abbreviations(text)
    protected, decimal_map = _protect_decimals(protected)
    protected, initial_map = _protect_initials(protected)

    # Phase 2: find sentence boundaries
    boundaries: list[int] = [0]

    for m in _SENTENCE_BOUNDARY.finditer(protected):
        boundaries.append(m.end())

    boundaries.append(len(protected))

    # Phase 3: extract raw sentences
    raw_sentences: list[str] = []
    for i in range(len(boundaries) - 1):
        s = protected[boundaries[i]:boundaries[i + 1]].strip()
        if s:
            raw_sentences.append(s)

    # Phase 4: restore protected patterns
    restored: list[str] = []
    for s in raw_sentences:
        s = _restore_abbreviations(s)
        s = _restore_decimals(s, decimal_map)
        s = _restore_initials(s, initial_map)
        restored.append(s)

    # Phase 5: clause-level splitting (semicolons / colons between clauses)
    result: list[str] = []
    for s in restored:
        result.extend(_split_clause_separators(s))

    # Phase 6: filter empties and strip
    return [s.strip() for s in result if s.strip()]


def _split_clause_separators(sentence: str) -> list[str]:
    """Split a sentence on semicolons/colons where they appear to separate
    complete clauses (preceding word length > 2)."""
    matches = list(_CLAUSE_SEPARATOR.finditer(sentence))
    if not matches:
        return [sentence]

    result: list[str] = []
    last_end = 0
    for m in matches:
        separator = m.group(2)
        # Split after the separator
        split_pos = m.start(2) + len(separator)
        result.append(sentence[last_end:split_pos].strip())
        last_end = split_pos

    # Append remaining text
    remainder = sentence[last_end:].strip()
    if remainder:
        result.append(remainder)

    if not result:
        return [sentence]
    return result


def split_long_sentence(sentence: str, max_chars: int = 600) -> list[str]:
    """Fallback split for sentences exceeding *max_chars*.

    Split points (in priority order):
    1. Semicolon + whitespace
    2. Colon + whitespace
    3. Comma + whitespace
    4. Em-dash / en-dash
    5. Any whitespace at nearest position to max_chars

    Returns sub-sentences; each sub-sentence guaranteed <= max_chars
    unless a single token exceeds max_chars.
    """
    if len(sentence) <= max_chars:
        return [sentence]

    result: list[str] = []
    remaining = sentence

    while len(remaining) > max_chars:
        # Try each split point in priority order
        split_pos = _find_best_split(remaining, max_chars)

        if split_pos <= 0 or split_pos > max_chars:
            # No good split found — hard cut at max_chars
            split_pos = max_chars

        result.append(remaining[:split_pos].strip())
        remaining = remaining[split_pos:].strip()

    if remaining:
        result.append(remaining)

    return result


def _find_best_split(text: str, max_chars: int) -> int:
    """Find the best split position in *text*, looking within [0, max_chars].

    Priority order (returns the *highest-priority* nearest to max_chars):
    1. Semicolon + whitespace
    2. Colon + whitespace
    3. Comma + whitespace
    4. Em-dash / en-dash (— or –)
    5. Any whitespace nearest to max_chars
    """
    window = text[:max_chars]

    # Priority 1: semicolon + space
    candidates = [m.end() for m in re.finditer(r";\s", window)]
    if candidates:
        return candidates[-1]

    # Priority 2: colon + space
    candidates = [m.end() for m in re.finditer(r":\s", window)]
    if candidates:
        return candidates[-1]

    # Priority 3: comma + space
    candidates = [m.end() for m in re.finditer(r",\s", window)]
    if candidates:
        return candidates[-1]

    # Priority 4: em-dash or en-dash
    candidates = [m.end() for m in re.finditer(r"[—–]\s?", window)]
    if candidates:
        return candidates[-1]

    # Priority 5: any whitespace nearest to max_chars
    candidates = [m.start() for m in re.finditer(r"\s", window)]
    if candidates:
        return candidates[-1]

    return max_chars
