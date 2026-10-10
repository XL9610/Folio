"""BGE WordPiece token counting — lightweight, no full model load.

bge-small-en-v1.5 silently truncates inputs at 512 WordPiece tokens. Counting
via the bundled tokenizer (not cl100k) lets the ingestion guard guarantee that
every indexed_text fits the model window without loading the embedding weights.
"""

from __future__ import annotations

import logging
import sys

from ...core.config import config

logger = logging.getLogger(__name__)

MAX_BGE_INPUT_TOKENS = 512

# Pin source-mode downloads to the same commit the packaging fetch uses
# (backend/scripts/fetch_embedding_model.py), so dev and bundled models match.
EMBEDDING_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

_tokenizer: object | None = None
_tokenizer_name = ""


def _get_tokenizer():
    global _tokenizer, _tokenizer_name
    name = config.embedding_model
    if _tokenizer is not None and _tokenizer_name == name:
        return _tokenizer
    from transformers import AutoTokenizer

    kwargs = {"local_files_only": getattr(sys, "frozen", False)}
    if not getattr(sys, "frozen", False):
        kwargs["revision"] = EMBEDDING_MODEL_REVISION
    _tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    _tokenizer_name = name
    return _tokenizer


def count_bge_tokens(text: str) -> int:
    """Token count of *text* as the model receives it (including special tokens)."""
    try:
        tok = _get_tokenizer()
    except Exception as exc:
        # Worst-case upper bound: WordPiece can emit one token per character for
        # punctuation/code/URLs, and BGE always adds [CLS]/[SEP] (+2), so
        # len(text) + 2 never undercounts.
        logger.warning("BGE tokenizer unavailable; using 1-char-1-token bound: %s", exc)
        return len(text) + 2
    return len(tok.encode(text, add_special_tokens=True))


def truncate_to_budget(
    text: str, budget: int = MAX_BGE_INPUT_TOKENS
) -> tuple[str, bool]:
    """Return *text* truncated to ≤ *budget* BGE tokens, plus a was_truncated flag."""
    try:
        tok = _get_tokenizer()
    except Exception as exc:
        # Fail-safe: with no tokenizer, truncate by the worst-case 1-char-1-token
        # bound, reserving two slots for BGE's [CLS]/[SEP].
        logger.warning("BGE tokenizer unavailable; truncating by char bound: %s", exc)
        if len(text) <= budget - 2:
            return text, False
        return text[: budget - 3] + "…", True

    ids = tok.encode(text, add_special_tokens=True)
    if len(ids) <= budget:
        return text, False
    # Drop [SEP] and the tail, then append "…". The margin keeps the re-encoded
    # result (with [CLS]/[SEP] re-added) within budget even if the ellipsis
    # round-trips to an extra token.
    return tok.decode(ids[: max(1, budget - 4)], skip_special_tokens=True) + "…", True
