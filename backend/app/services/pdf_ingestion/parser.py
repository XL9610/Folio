"""PDF text extraction using PyMuPDF."""

from __future__ import annotations

import re

import fitz  # PyMuPDF

from .models import DocumentPage

# ---------------------------------------------------------------------------
# Title extraction helpers
# ---------------------------------------------------------------------------

# Patterns to reject as display titles (filenames, ISBNs, etc.)
_REJECT_TITLE_PATTERNS = [
    re.compile(r"^\d+\.pdf$", re.IGNORECASE),
    re.compile(r"^\d{10,}$"),  # ISBN-like
    re.compile(r"^\d{3}[-.]?\d{1,5}[-.]?\d{1,7}[-.]?\d{1,7}[-.]?\d$"),  # ISBN with hyphens/dots
    re.compile(r"^[\d\-\s]{13,}$"),  # long digit/hyphen strings
    re.compile(r"^untitled", re.IGNORECASE),
    re.compile(r"^microsoft word", re.IGNORECASE),
    re.compile(r"\.pdf$", re.IGNORECASE),  # ends in .pdf
    re.compile(r"\.epub$", re.IGNORECASE),
    re.compile(r"\.djvu$", re.IGNORECASE),
    re.compile(r"^[a-z0-9_\-]{20,}\.pdf$", re.IGNORECASE),  # hash-style filenames
    re.compile(r"^fulltext", re.IGNORECASE),
]

# Title line detection — simple heuristic: most words start with capital letter
def _looks_like_title_line(line: str) -> bool:
    """Check if a line looks like a book title (most words capitalized)."""
    t = line.strip()
    if not t or len(t) < 8:
        return False
    words = [w for w in t.split() if len(w) >= 2]
    if len(words) < 2:
        return False
    # Count capitalized words (first letter uppercase)
    capped = sum(1 for w in words if w[0].isupper())
    if capped == 0:
        return False
    # Most words should be capitalized, or it starts with article + has capped words
    if capped >= len(words) * 0.5 or (capped >= 2 and words[0].lower() in ("the", "a", "an")):
        return True
    return False

# Lines that are clearly NOT title material (publisher boilerplate, series info, etc.)
_NON_TITLE_LINE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^oxford\s+history\s+of\s+art$", re.IGNORECASE),
    re.compile(r"^titles?\s+in\s+the\s+oxford", re.IGNORECASE),
    re.compile(r"^w\.?\s*w\.?\s*norton", re.IGNORECASE),
    re.compile(r"^new\s+york\s*.*london", re.IGNORECASE),
    re.compile(r"^published\s+by", re.IGNORECASE),
    re.compile(r"^all\s+rights?\s+reserved", re.IGNORECASE),
    re.compile(r"^\d{4}\s*$"),  # standalone year
    re.compile(r"^edition$", re.IGNORECASE),
    re.compile(r"^edited\s+by", re.IGNORECASE),
    re.compile(r"^translated\s+by", re.IGNORECASE),
    re.compile(r"^illustrated\s+by", re.IGNORECASE),
    re.compile(r"^with\s+\d+\s+illustrations", re.IGNORECASE),
    # Publisher addresses / city lists
    re.compile(r".*\b(?:street|avenue|road|lane|drive|boulevard|place|square)\b", re.IGNORECASE),
    re.compile(r"^(?:oxford|new\s+york|london|paris|berlin|tokyo|sydney|beijing|shanghai|mumbai|delhi|toronto|boston|chicago|los\s+angeles|san\s+francisco)\s", re.IGNORECASE),
    re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){4,}$"),  # 5+ capitalized words (likely address/catalog)
]

# Piracy/trash patterns to strip from extracted titles
_PIRACY_PATTERNS_RE = re.compile(
    r"(?:_?z[_-]?library[_-]?sk[_,.\s]*|_?1lib[_-]?sk[_,.\s]*|\(z[_-]?lib[^)]*\)|\[z[_-]?lib[^\]]*\])",
    re.IGNORECASE,
)


def _clean_title_text(text: str) -> str:
    """Remove piracy suffixes and clean whitespace."""
    s = _PIRACY_PATTERNS_RE.sub("", text)
    s = re.sub(r"[\s,;.\-]+$", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _is_good_title(text: str) -> bool:
    """Check if extracted text looks like a real title (not a filename/ISBN)."""
    t = text.strip()
    if not t or len(t) < 5:
        return False
    for pat in _REJECT_TITLE_PATTERNS:
        if pat.match(t):
            return False
    # Must have at least one space (real title vs single word)
    if " " not in t and len(t) > 30:
        # Single long word — probably not a title
        return False
    return True


def _is_title_candidate(line: str) -> bool:
    """Check if a line looks like a real title candidate (not publisher boilerplate)."""
    if not line or len(line) < 5:
        return False
    for pat in _NON_TITLE_LINE_PATTERNS:
        if pat.match(line.strip()):
            return False
    if re.match(r"^\d{1,4}$", line.strip()):
        return False
    return _looks_like_title_line(line.strip())


def _score_title_candidate(text: str, position: int = 0) -> float:
    """Score a title candidate. Higher = more likely to be the real title.

    Factors:
      - Length (3-8 words ideal, very short or very long = penalty)
      - Position on page (near top = higher)
      - Contains colon/subtitle separator (+)
      - Starts with common title words (+)
      - Doesn't look like an address or catalog listing (+)
    """
    score = 0.0
    words = len(text.split())
    if 3 <= words <= 8:
        score += 4.0
    elif 2 <= words <= 12:
        score += 2.0
    elif words >= 13:
        score -= 3.0  # probably an address or catalog line
    else:
        score += 0.5

    # Near top of page
    if position < 5:
        score += 2.0
    elif position < 10:
        score += 1.0

    # Has subtitle separator (colon, dash)
    if re.search(r"[:;–—]", text):
        score += 2.0

    # Starts with common title words
    if re.match(r"^(the|a|an|part|chapter)\s", text, re.IGNORECASE):
        score += 1.0

    # Penalize lines that look like addresses/city lists (many commas)
    comma_count = text.count(",")
    if comma_count >= 3:
        score -= 4.0
    elif comma_count >= 2:
        score -= 2.0

    # Penalize if too many digits
    digit_ratio = sum(1 for ch in text if ch.isdigit()) / max(len(text), 1)
    if digit_ratio > 0.3:
        score -= 3.0
    elif digit_ratio > 0.1:
        score -= 1.0

    # Bonus for likely subtitle separator with short subtitle
    parts = re.split(r"[:;–—]", text, maxsplit=1)
    if len(parts) == 2:
        subtitle_words = len(parts[1].split())
        if 2 <= subtitle_words <= 5:
            score += 1.5

    return score


def _extract_candidates_from_pages(doc, max_pages: int = 5, filename: str = "") -> list[tuple[str, float]]:
    """Extract title candidates from first pages of a PDF document.

    Strategy: try individual lines and joined pairs from the first few pages.
    """
    candidates: list[tuple[str, float]] = []

    # Extract filename keywords for boosting
    filename_words: set[str] = set()
    if filename:
        clean = re.sub(r"[^a-zA-Z\s]", " ", filename).lower()
        filename_words = {w for w in clean.split() if len(w) >= 4}

    for page_num in range(min(max_pages, doc.page_count)):
        text = doc[page_num].get_text()
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        for i in range(min(len(lines), 25)):
            line = lines[i]

            # Single-line candidate
            if _is_title_candidate(line):
                score = _score_title_candidate(line, i)
                score += _filename_boost(line, filename_words)
                candidates.append((_clean_title_text(line), score))

            # Join with next line (handles broken titles)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                # Join if both look like title fragments
                if line.rstrip().endswith(":"):
                    combined = f"{line.rstrip()[:-1]}: {next_line}"  # strip trailing colon, add back once
                else:
                    combined = f"{line} {next_line}"
                if _is_title_candidate(combined) and len(combined.split()) <= 12:
                    score = _score_title_candidate(combined, i)
                    score += _filename_boost(combined, filename_words) + 1.0  # bonus for joining
                    candidates.append((_clean_title_text(combined), score))

    candidates.sort(key=lambda x: -x[1])
    return candidates


def _filename_boost(text: str, filename_words: set[str]) -> float:
    """Boost score if candidate contains words from the filename."""
    if not filename_words:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for w in filename_words if w in text_lower)
    return min(matches * 1.5, 6.0)  # cap at 6.0


def extract_display_title(file_path: str, filename: str = "") -> str:
    """Extract a human-readable display title from a PDF.

    Priority:
      1. PDF metadata title (if meaningful, not a filename/ISBN)
      2. First few pages' text heuristics (scored candidates)
      3. Cleaned filename fallback
    """
    try:
        doc = fitz.open(file_path)
    except (FileNotFoundError, RuntimeError):
        return _fallback_title(filename)

    try:
        # 1. Try PDF metadata title
        meta = doc.metadata or {}
        meta_title = (meta.get("title") or "").strip()
        if _is_good_title(meta_title):
            return _clean_title_text(meta_title)

        # 2. Heuristic from first 5 pages — scored candidates
        candidates = _extract_candidates_from_pages(doc, max_pages=5, filename=filename)
        if candidates:
            best = candidates[0][0]
            if _is_good_title(best):
                return best

        # 3. Fallback to cleaned filename
        return _fallback_title(filename)
    finally:
        doc.close()


def _fallback_title(filename: str) -> str:
    """Clean a filename into a display title."""
    if not filename:
        return ""
    s = filename.strip()
    # Remove extension
    s = re.sub(r"\.(pdf|epub|mobi|djvu)$", "", s, flags=re.IGNORECASE)
    # Remove piracy markers
    s = _PIRACY_PATTERNS_RE.sub("", s)
    # Replace underscores
    s = s.replace("_", " ")
    # Remove trailing garbage
    s = re.sub(r"[,;\s\-]+$", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


class PDFParser:
    """Extracts text and metadata from a PDF file."""

    def parse(self, file_path: str) -> list[DocumentPage]:
        """Parse a PDF file and return a list of DocumentPage objects.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of DocumentPage with extracted text and page metadata.
            Returns empty list if file is not found or can't be read.
        """
        try:
            doc = fitz.open(file_path)
        except (FileNotFoundError, RuntimeError):
            return []

        try:
            pages: list[DocumentPage] = []
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                pages.append(DocumentPage(
                    page_number=page_num + 1,  # 1-indexed for user-facing
                    text=text,
                    metadata={
                        "width": page.rect.width,
                        "height": page.rect.height,
                    },
                ))
            return pages
        finally:
            doc.close()
