"""Sentence-boundary-aware chunking for unstructured documents.

Strategy: ``sentence_boundary_layout_hybrid``

Algorithm:
1. Per page: split_sentences() -> preserve (page_number, sentence) mapping
2. Greedy merge sentences within each page until approaching target_tokens
3. Single sentence > hard_max -> split_long_sentence()
4. Overlap: last overlap_tokens tokens from chunk N become prefix of chunk N+1.
   Overlap at sentence boundary.
5. Page metadata: min(page_start), max(page_end)
6. Table/caption detection: >=3 consecutive numeric tokens OR starts with
   "Table", "Figure", "Fig.", "Exhibit" -> has_table_or_caption: true
7. Chunk metadata: chunking_strategy, outline_source, token_count,
   has_table_or_caption, sentence_count
"""

from __future__ import annotations

import re
import tiktoken

from ..models import Chunk, ChunkingResult, DocumentPage
from ..sentence_splitter import split_long_sentence, split_sentences

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TABLE_CAPTION_TRIGGERS: tuple[str, ...] = ("Table", "Figure", "Fig.", "Exhibit")
_NUMERIC_TOKEN_RE: re.Pattern[str] = re.compile(
    r'^[\d,.]{2,}$',
)


def _has_table_or_caption(text: str) -> bool:
    """Detect if *text* likely contains a table or figure caption.

    Heuristics:
    - Starts with "Table", "Figure", "Fig.", or "Exhibit"
    - Contains >=3 consecutive numeric tokens (table-like data)
    """
    stripped = text.strip()

    # Check for caption-like opening
    for trigger in _TABLE_CAPTION_TRIGGERS:
        if stripped.startswith(trigger):
            return True

    # Check for >=3 consecutive numeric tokens
    tokens = stripped.split()
    consecutive_numeric = 0
    for token in tokens:
        if _NUMERIC_TOKEN_RE.match(token):
            consecutive_numeric += 1
            if consecutive_numeric >= 3:
                return True
        else:
            consecutive_numeric = 0

    return False


def chunk_layout_hybrid(
    pages: list[DocumentPage],
    doc_id: str,
    *,
    target_tokens: int = 380,
    hard_max_tokens: int = 460,
    overlap_tokens: int = 40,
    tokenizer_name: str = "cl100k_base",
) -> ChunkingResult:
    """Sentence-boundary-aware chunking for unstructured documents.

    Args:
        pages: List of DocumentPage objects from PDF parsing.
        doc_id: Stable document identifier.
        target_tokens: Ideal token count per chunk (default 380).
        hard_max_tokens: Maximum token count before fallback splitting (default 460).
        overlap_tokens: Number of tokens to overlap between adjacent chunks (default 40).
        tokenizer_name: tiktoken encoding name (default "cl100k_base").

    Returns:
        ChunkingResult with ``strategy_used="layout_hybrid"``.
    """
    enc = tiktoken.get_encoding(tokenizer_name)

    # Step 1: Split each page into sentences, preserving (page_number, sentence)
    sentence_entries: list[tuple[int, str]] = []
    for page in pages:
        sentences = split_sentences(page.text)
        for sentence in sentences:
            sentence_entries.append((page.page_number, sentence))

    if not sentence_entries:
        return ChunkingResult(
            document_id=doc_id,
            pages=pages,
            strategy_used="layout_hybrid",
            warnings=["No sentences extracted from pages."],
        )

    # Step 2 & 3: Greedy merge sentences into chunks, enforcing token budget
    raw_chunks: list[_RawChunk] = []
    current_sentences: list[str] = []
    current_tokens: list[int] = []
    current_token_count: int = 0
    current_page_start: int | None = None
    current_page_end: int = 1

    def _flush_chunk() -> None:
        nonlocal current_sentences, current_tokens, current_token_count
        nonlocal current_page_start, current_page_end

        if not current_sentences:
            return

        text = " ".join(current_sentences)
        raw_chunks.append(_RawChunk(
            text=text,
            tokens=current_tokens,
            token_count=current_token_count,
            page_start=current_page_start or 1,
            page_end=current_page_end,
            sentence_count=len(current_sentences),
        ))

        current_sentences = []
        current_tokens = []
        current_token_count = 0
        current_page_start = None
        current_page_end = 1

    for page_num, sentence in sentence_entries:
        sent_tokens = enc.encode(sentence)
        sent_token_count = len(sent_tokens)

        if sent_token_count > hard_max_tokens:
            # Step 3: single sentence exceeds hard max — force-split
            if current_sentences:
                _flush_chunk()

            sub_sentences = split_long_sentence(sentence)
            for sub in sub_sentences:
                sub_tokens = enc.encode(sub)
                sub_count = len(sub_tokens)
                if sub_count > hard_max_tokens:
                    # Super-long single token sequence — chunk by hard tokens.
                    # Use the smaller of target_tokens and hard_max_tokens
                    # as the step so we reliably split long sub-sentences.
                    step = min(target_tokens, hard_max_tokens)
                    for i in range(0, len(sub_tokens), step):
                        seg = sub_tokens[i:i + step]
                        seg_text = enc.decode(seg)
                        raw_chunks.append(_RawChunk(
                            text=seg_text,
                            tokens=seg,
                            token_count=len(seg),
                            page_start=page_num,
                            page_end=page_num,
                            sentence_count=1,
                        ))
                else:
                    # Treat each sub-sentence as its own chunk
                    raw_chunks.append(_RawChunk(
                        text=sub,
                        tokens=sub_tokens,
                        token_count=sub_count,
                        page_start=page_num,
                        page_end=page_num,
                        sentence_count=1,
                    ))
            continue

        # Can we add this sentence to the current chunk?
        if current_token_count + sent_token_count <= target_tokens:
            current_sentences.append(sentence)
            current_tokens.extend(sent_tokens)
            current_token_count += sent_token_count
            if current_page_start is None:
                current_page_start = page_num
            current_page_end = page_num
        else:
            # Flush current chunk and start new one
            _flush_chunk()
            current_sentences.append(sentence)
            current_tokens = list(sent_tokens)
            current_token_count = sent_token_count
            current_page_start = page_num
            current_page_end = page_num

    # Flush any remaining sentences
    _flush_chunk()

    # Step 4: Apply overlap — last overlap_tokens tokens from chunk N-1
    # become the prefix of chunk N, snapped to a sentence boundary.
    chunks_with_overlap: list[_RawChunk] = _apply_overlap(
        raw_chunks, enc, overlap_tokens
    )

    # Step 5 & 6 & 7: Build final Chunk objects with metadata
    final_chunks: list[Chunk] = []
    for i, rc in enumerate(chunks_with_overlap):
        meta: dict = {
            "chunking_strategy": "sentence_boundary_layout_hybrid",
            "outline_source": "page_window",
            "token_count": rc.token_count,
            "has_table_or_caption": _has_table_or_caption(rc.text),
            "sentence_count": rc.sentence_count,
        }

        final_chunks.append(Chunk(
            text=rc.text,
            document_id=doc_id,
            chunk_index=i,
            page_start=rc.page_start,
            page_end=rc.page_end,
            chunk_strategy="sentence_boundary_layout_hybrid",
            metadata=meta,
        ))

    return ChunkingResult(
        document_id=doc_id,
        pages=pages,
        chunks=final_chunks,
        strategy_used="layout_hybrid",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RawChunk:
    """Intermediate representation before overlap and final metadata."""
    __slots__ = ("text", "tokens", "token_count", "page_start", "page_end", "sentence_count")

    def __init__(
        self,
        text: str,
        tokens: list[int],
        token_count: int,
        page_start: int,
        page_end: int,
        sentence_count: int,
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.token_count = token_count
        self.page_start = page_start
        self.page_end = page_end
        self.sentence_count = sentence_count


def _apply_overlap(
    raw_chunks: list[_RawChunk],
    enc: tiktoken.Encoding,
    overlap_tokens: int,
) -> list[_RawChunk]:
    """Apply sentence-boundary overlap between adjacent chunks.

    The last *overlap_tokens* tokens from chunk N are prepended to chunk N+1,
    snapped to the nearest sentence boundary so we never split mid-sentence
    in the overlap prefix.
    """
    if overlap_tokens <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    result: list[_RawChunk] = [raw_chunks[0]]

    for i in range(1, len(raw_chunks)):
        prev = raw_chunks[i - 1]
        curr = raw_chunks[i]

        # Take the last *overlap_tokens* from prev, then snap to a sentence end
        overlap_tok = prev.tokens[-overlap_tokens:] if len(prev.tokens) >= overlap_tokens else prev.tokens
        overlap_text = enc.decode(overlap_tok)

        # Snap to last sentence boundary in overlap text
        snapped = _snap_to_sentence_end(overlap_text)
        if snapped:
            prefix = snapped
        else:
            prefix = overlap_text

        # Build the overlapped text: overlap prefix + current text
        new_text = prefix + " " + curr.text if prefix.strip() else curr.text
        new_tokens = enc.encode(new_text)
        new_token_count = len(new_tokens)

        # Update current chunk
        curr.text = new_text
        curr.tokens = new_tokens
        curr.token_count = new_token_count
        curr.sentence_count = curr.sentence_count + _count_sentences_in_text(prefix)

        result.append(curr)

    return result


def _snap_to_sentence_end(text: str) -> str:
    """Return the portion of *text* up to and including the last
    sentence-ending boundary.  Returns empty string if no boundary found."""
    # Walk backwards to find the last sentence-ending character
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".!?":
            # Check if it's a real sentence end (followed by space/newline or end)
            if i + 1 >= len(text) or text[i + 1] in " \n\r\t\"')]}:":
                after_punc = text[i + 1:].lstrip()
                if not after_punc or after_punc[0].isupper() or after_punc[0].isdigit():
                    return text[:i + 1]

    return ""


def _count_sentences_in_text(text: str) -> int:
    """Count sentences in *text* by looking for sentence-ending punctuation."""
    if not text.strip():
        return 0
    count = 0
    for i, ch in enumerate(text):
        if ch in ".!?":
            if i + 1 >= len(text) or text[i + 1] in " \n\r\t\"')]}:":
                count += 1
    return max(count, 1) if text.strip() else 0
