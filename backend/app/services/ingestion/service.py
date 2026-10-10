"""Ingestion service — orchestrate extraction, chunking, persistence, embedding, indexing.

This is the main pipeline that ties together:
  - ExtractorRegistry (content extraction)
  - ExtractionArtifactStore (durable extraction checkpoints)
  - ChunkerSelector (structure-aware chunking)
  - SQLite repositories (metadata + FTS)
  - SentenceTransformerEmbedder (dense vectors)
  - SqliteVecStore (dense vector search)
  - BM25x keyword backend
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.config import config
from ...repositories import chunk_repository, document_repository
from ...repositories.extraction_repository import upsert_document_extraction
from ..chunking.selector import ChunkerSelector
from ..embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from ..extraction.artifact_store import ExtractionArtifactStore
from ..extraction.models import ExtractedDocument, SourceInput
from ..extraction.pdf import PdfTextbookExtractor, is_text_poor_pdf
from ..extraction.registry import ExtractorRegistry, build_default_extractor_registry
from ..index.sqlite_vec_store import SqliteVecStore
from ..llm import LLMConfig
from ..pdf_ingestion.chunker import PDFChunker
from ..pdf_ingestion.structure_metadata import (
    build_structure_metadata,
    document_structure_fields,
)
from ..toc_llm_classifier.classifier import LLMTOCClassifier
from ..embedding.bge_tokenizer import truncate_to_budget
from .index_text import build_index_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class IngestionResult:
    """Result of ingesting a single source document."""

    document_id: str
    notebook_id: str
    filename: str
    display_title: str = ""
    source_type: str = "pdf"
    extraction_engine: str = ""
    page_count: int = 0
    chunk_count: int = 0
    status: str = "completed"
    warnings: list[str] = field(default_factory=list)
    # Chunking metadata
    strategy_used: str = "unknown"
    chunking_mode: str = ""
    first_chunk_title: str = ""
    front_matter_skipped: bool = False
    toc_entries_total: int = 0
    toc_entries_used: int = 0
    toc_filter_strategy: str = "rules"
    llm_toc_filter_used: bool = False
    # Body-window metadata
    body_start_title: str = ""
    body_end_title: str = ""
    front_matter_skipped_count: int = 0
    back_matter_skipped_count: int = 0
    wrapper_skipped_count: int = 0
    # Structure reliability metadata
    structure_status: str = "unknown_legacy"
    embedded_toc_present: bool | None = None
    synthetic_outline_mode: str | None = None
    structure_confidence: str | None = None
    structure_user_note: str | None = None
    structure_llm_instruction: str | None = None


# ---------------------------------------------------------------------------
# Public API — PDF compatibility adapter
# ---------------------------------------------------------------------------


def ingest_pdf(
    file_path: str,
    filename: str,
    notebook_id: str,
    use_llm_toc_filter: bool = config.chunking_llm_toc_filter,
    force_reingest: bool = False,
    chunker: PDFChunker | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    dense_store: SqliteVecStore | None = None,
) -> IngestionResult:
    """Compatibility adapter — delegates PDF ingestion through ingest_source.

    Preserves existing call-site compatibility. All parameters are forwarded.
    """
    source = SourceInput(
        source_type="pdf",
        file_path=file_path,
        filename=filename,
        notebook_id=notebook_id,
    )
    return ingest_source(
        source=source,
        use_llm_toc_filter=use_llm_toc_filter,
        force_reingest=force_reingest,
        pdf_chunker=chunker,
        embedder=embedder,
        dense_store=dense_store,
    )


def _build_pdf_chunker(use_llm_toc_filter: bool) -> PDFChunker:
    """Build the default PDFChunker, wiring the LLM TOC classifier when enabled.

    Mirrors pdf_router: the classifier is built only when the flag is on and the
    LLM is configured; otherwise the chunker falls back to rules-based filtering.
    """
    toc_llm_classifier: LLMTOCClassifier | None = None
    if use_llm_toc_filter:
        llm_config = LLMConfig()
        if llm_config.is_configured:
            toc_llm_classifier = LLMTOCClassifier(llm_config)
    return PDFChunker(toc_llm_classifier=toc_llm_classifier)


# ---------------------------------------------------------------------------
# Public API — source-agnostic ingestion
# ---------------------------------------------------------------------------


def ingest_source(
    source: SourceInput,
    use_llm_toc_filter: bool = config.chunking_llm_toc_filter,
    force_reingest: bool = False,
    extractor_registry: ExtractorRegistry | None = None,
    artifact_store: ExtractionArtifactStore | None = None,
    chunker_selector: ChunkerSelector | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    dense_store: SqliteVecStore | None = None,
    pdf_chunker: PDFChunker | None = None,
    extracted_document: ExtractedDocument | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> IngestionResult:
    """Run the full ingestion pipeline on any supported source.

    Steps:
      1. Select and run extractor
      2. PDF fallback check (text-poor PDF → Kreuzberg)
      3. Duplicate detection (notebook + content hash)
      4. Create source_document row
      5. Persist extraction artifacts
      6. Chunk
      7. Insert chunks into SQLite
      8. Embed + index in sqlite-vec
      9. Sync to BM25x keyword index
    """
    _registry = extractor_registry or build_default_extractor_registry()
    _artifact = artifact_store or ExtractionArtifactStore()
    _chunk_sel = chunker_selector or ChunkerSelector(
        pdf_chunker=pdf_chunker or _build_pdf_chunker(use_llm_toc_filter)
    )
    _embedder = embedder or SentenceTransformerEmbedder()
    _dense = dense_store or SqliteVecStore()

    # --- Step 1: Extract (skip if pre-extracted document provided) ---
    import asyncio

    extracted: ExtractedDocument
    warnings: list[str]
    selection: Any = None

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if extracted_document is not None:
        extracted = extracted_document
        warnings = list(extracted.warnings)
    else:
        selection = _registry.select(source)

        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, selection.extractor.extract(source))
                extracted = future.result(timeout=300)
        else:
            extracted = asyncio.run(selection.extractor.extract(source))

        warnings = list(extracted.warnings)

    # --- Step 1b: PDF fallback check ---
    fallback_reason = ""
    if (
        extracted_document is None
        and isinstance(selection.extractor, PdfTextbookExtractor)
        and extracted.pages
        and is_text_poor_pdf(extracted.pages)
    ):
        from ..extraction.kreuzberg_fallback import KreuzbergFallbackExtractor
        fb = KreuzbergFallbackExtractor()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, fb.extract(source))
                extracted = future.result(timeout=300)
        else:
            extracted = asyncio.run(fb.extract(source))
        fallback_reason = "PDF text too sparse; used Kreuzberg fallback."
        warnings.extend(extracted.warnings)
        warnings.append(fallback_reason)

    if progress_callback:
        progress_callback("extracted", 30)

    # --- Step 2: Content hash for dedup ---
    content_hash = compute_extracted_content_hash(extracted)

    # --- Step 3: Dedup check ---
    existing = document_repository.get_document_by_hash(
        source.notebook_id, content_hash,
    )
    if existing:
        if existing.get("status") == "failed":
            # Ghost left by a previous failed attempt — free the content hash and
            # retry instead of returning "already_ingested".
            logger.info(
                "Retrying previously-failed document (hash=%s, existing=%s).",
                content_hash[:12], existing["id"],
            )
            try:
                _dense.delete_by_document_id(existing["id"])
            except Exception as exc:
                logger.error("Dense cleanup failed during retry for %s: %s", existing["id"], exc)
                raise
            try:
                _artifact.delete(existing["id"])
            except Exception as exc:
                logger.error("Artifact cleanup failed during retry for %s: %s", existing["id"], exc)
                raise
            document_repository.delete_document(existing["id"])
        elif not force_reingest:
            existing_structure = document_structure_fields(existing)
            logger.info(
                "Duplicate document detected (hash=%s, existing=%s), returning existing.",
                content_hash[:12], existing["id"],
            )
            return IngestionResult(
                document_id=existing["id"],
                notebook_id=source.notebook_id,
                filename=source.filename,
                display_title=existing.get("display_title", ""),
                source_type=source.source_type,
                extraction_engine=extracted.extraction_engine,
                page_count=existing["page_count"],
                chunk_count=existing["chunk_count"],
                status="already_ingested",
                warnings=[
                    "Document already ingested. Use force_reingest=true to re-ingest."
                ],
                strategy_used="unknown",
                chunking_mode=existing_structure.get("chunking_mode") or "",
                first_chunk_title="",
                front_matter_skipped=False,
                toc_entries_total=0,
                toc_entries_used=0,
                toc_filter_strategy="rules",
                llm_toc_filter_used=False,
                body_start_title="",
                body_end_title="",
                front_matter_skipped_count=0,
                back_matter_skipped_count=0,
                wrapper_skipped_count=0,
                structure_status=existing_structure.get("structure_status", "unknown_legacy"),
                embedded_toc_present=existing_structure.get("embedded_toc_present"),
                synthetic_outline_mode=existing_structure.get("synthetic_outline_mode"),
                structure_confidence=existing_structure.get("structure_confidence"),
                structure_user_note=existing_structure.get("structure_user_note"),
                structure_llm_instruction=existing_structure.get("structure_llm_instruction"),
            )
        else:
            logger.info(
                "Force re-ingesting document (hash=%s), deleting existing %s.",
                content_hash[:12], existing["id"],
            )
            # Must delete dense vectors first, then SQLite — otherwise
            # stale vectors pollute retrieval.
            try:
                _dense.delete_by_document_id(existing["id"])
            except Exception as exc:
                logger.error("Dense cleanup failed during force_reingest for %s: %s", existing["id"], exc)
                raise
            try:
                _artifact.delete(existing["id"])
            except Exception as exc:
                logger.error("Artifact cleanup failed during force_reingest for %s: %s", existing["id"], exc)
                raise
            document_repository.delete_document(existing["id"])

    # --- Step 4: Display title (extractor already computed the PDF title) ---
    display_title = extracted.title or source.filename

    # --- Step 5: Create source document ---
    doc = document_repository.create_source_document(
        notebook_id=source.notebook_id,
        filename=source.filename,
        display_title=display_title,
        page_count=len(extracted.pages) if extracted.pages else 0,
        content_hash=content_hash,
    )
    doc_id = doc["id"]

    # --- Step 6: Persist extraction artifacts ---
    try:
        paths = _artifact.save(doc_id, extracted)
        metadata_json = json.dumps(
            {
                "source_type": source.source_type,
                "original_filename": source.filename,
                **extracted.metadata,
            },
            ensure_ascii=False,
        )
        upsert_document_extraction(
            document_id=doc_id,
            extraction_engine=extracted.extraction_engine,
            full_text_path=str(paths.content_txt),
            markdown_path=str(paths.content_md) if paths.content_md else None,
            pages_path=str(paths.pages_jsonl) if paths.pages_jsonl else None,
            metadata_json=metadata_json,
            warnings_json=json.dumps(warnings, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("Artifact persistence failed for %s: %s", doc_id, exc)
        warnings.append(f"Artifact persistence failed: {exc}")

    # --- Steps 7-8: Chunk and insert. Any failure here must not leave a
    # half-ingested 'pending' ghost that blocks re-upload via the content hash —
    # mark the document failed and re-raise so the job records the error.
    try:
        chunking_result = _chunk_sel.chunk(
            extracted,
            document_id=doc_id,
            source_file_path=source.file_path if source.source_type == "pdf" else None,
        )

        if not chunking_result.chunks:
            raise ValueError("No extractable content from source. Possibly empty or unsupported.")

        structure = build_structure_metadata(chunking_result) if source.source_type == "pdf" else None

        # --- Step 8: Insert chunks into SQLite (with context enrichment) ---
        chunk_dicts: list[dict[str, Any]] = []
        for c in chunking_result.chunks:
            chunk_dicts.append(chunk_to_insert_dict(c, source.source_type))

        # Phase 1XA: enrich each chunk dict with deterministic context prefix,
        # then enforce the embedding window against the bundled BGE tokenizer so
        # no indexed_text is silently truncated by the model at 512 WordPiece.
        indexed_texts: list[str] = []
        for d in chunk_dicts:
            result = build_index_text(display_title, d.get("section_path", ""), d["text"])
            d["context_prefix"] = result.context_prefix
            d["context_mode"] = result.context_mode
            fitted, truncated = truncate_to_budget(result.indexed_text)
            if truncated:
                warnings.append(
                    "Chunk exceeded the 512-token embedding window and was "
                    "truncated before indexing"
                )
            indexed_texts.append(fitted)

        inserted = chunk_repository.insert_chunks(doc_id, chunk_dicts)
        document_repository.update_document_status(doc_id, "indexing", chunk_count=inserted)
        if progress_callback:
            progress_callback("indexing", 70)

        # Phase 1XA: fetch persisted chunks and validate count before indexing
        db_chunks = chunk_repository.get_chunks_by_document(doc_id)
        if len(db_chunks) != len(indexed_texts):
            raise RuntimeError(
                f"Chunk count mismatch after insert: "
                f"db={len(db_chunks)} indexed_texts={len(indexed_texts)}"
            )
    except Exception:
        # Clear any chunks committed before the failure so a "failed" document
        # never carries stale chunks/FTS rows into a later re-upload.
        try:
            chunk_repository.delete_chunks_by_document(doc_id)
        except Exception as exc:
            logger.warning("Failed to clear chunks for %s: %s", doc_id, exc)
        document_repository.update_document_status(doc_id, "failed")
        raise

    # --- Step 9: Embed and index ---
    texts = indexed_texts
    dense_healthy, dense_msg, _dense_ms = _dense.health_check(timeout=3.0)
    final_status: str

    if not dense_healthy:
        logger.warning(
            "Dense index unavailable for %s: %s. Skipping dense indexing.", doc_id, dense_msg,
        )
        document_repository.update_document_status(doc_id, "fts_only", chunk_count=inserted)
        final_status = "fts_only"
        warnings.append(f"Dense index unavailable (sqlite-vec): {dense_msg}")
    else:
        try:
            vectors = _embedder.embed_texts(texts)
            if not vectors:
                raise ValueError("Embedder returned empty vectors")

            vector_size = len(vectors[0])
            _dense.ensure_collection(vector_size)

            dense_payloads = []
            for dbc, _vec in zip(db_chunks, vectors, strict=False):
                dense_payloads.append({
                    "chunk_id": dbc["id"],
                    "document_id": doc_id,
                    "notebook_id": source.notebook_id,
                    "title": dbc["title"],
                    "section_path": dbc["section_path"],
                    "page_start": dbc["page_start"],
                    "page_end": dbc["page_end"],
                    "chunk_strategy": dbc["chunk_strategy"],
                    "source_type": source.source_type,
                })

            _dense.upsert_chunks(dense_payloads, vectors)
            document_repository.update_document_status(doc_id, "completed", chunk_count=inserted)
            final_status = "completed"
        except Exception as exc:
            logger.warning(
                "Dense indexing failed for %s: %s. Marking as partial.", doc_id, exc,
            )
            document_repository.update_document_status(doc_id, "indexed_partial", chunk_count=inserted)
            final_status = "indexed_partial"
            warnings.append(f"Dense index unavailable: {exc}")

    # --- Step 10: BM25x keyword sync ---
    try:
        from ..retrieval.keyword_backend import get_keyword_backend
        kw = get_keyword_backend()
        if db_chunks:
            kw.index_chunks([
                {
                    "chunk_id": dbc["id"],
                    "document_id": dbc["document_id"],
                    "text": dbc["text"],
                    "title": dbc.get("title", ""),
                    "section_path": dbc.get("section_path", ""),
                    "page_start": dbc.get("page_start", -1),
                    "page_end": dbc.get("page_end", -1),
                    "indexed_text": idx_text,
                }
                for dbc, idx_text in zip(db_chunks, indexed_texts, strict=True)
            ])
    except Exception as exc:
        logger.debug("BM25x sync skipped or failed: %s", exc)

    if progress_callback:
        progress_callback("done", 100)

    # --- Build result ---
    return IngestionResult(
        document_id=doc_id,
        notebook_id=source.notebook_id,
        filename=source.filename,
        display_title=display_title,
        source_type=source.source_type,
        extraction_engine=extracted.extraction_engine,
        page_count=len(extracted.pages) if extracted.pages else 0,
        chunk_count=inserted,
        status=final_status,
        warnings=warnings,
        strategy_used=chunking_result.strategy_used if chunking_result.chunks else "unknown",
        chunking_mode=structure.chunking_mode if structure else chunking_result.chunking_mode,
        first_chunk_title=chunking_result.first_chunk_title,
        front_matter_skipped=chunking_result.front_matter_skipped,
        toc_entries_total=chunking_result.toc_entries_total,
        toc_entries_used=chunking_result.toc_entries_used,
        toc_filter_strategy=chunking_result.toc_filter_strategy,
        llm_toc_filter_used=chunking_result.llm_toc_filter_used,
        body_start_title=chunking_result.body_start_title,
        body_end_title=chunking_result.body_end_title,
        front_matter_skipped_count=chunking_result.front_matter_skipped_count,
        back_matter_skipped_count=chunking_result.back_matter_skipped_count,
        wrapper_skipped_count=chunking_result.wrapper_skipped_count,
        structure_status=structure.structure_status if structure else "unknown_legacy",
        embedded_toc_present=structure.embedded_toc_present if structure else None,
        synthetic_outline_mode=structure.synthetic_outline_mode if structure else None,
        structure_confidence=structure.structure_confidence if structure else None,
        structure_user_note=structure.structure_user_note if structure else None,
        structure_llm_instruction=structure.structure_llm_instruction if structure else None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_extracted_content_hash(document: ExtractedDocument) -> str:
    """Compute a SHA-256 hash of extracted full_text for dedup."""
    return hashlib.sha256(document.full_text.encode("utf-8")).hexdigest()


def chunk_to_insert_dict(chunk: Any, source_type: str) -> dict[str, Any]:
    """Map a Chunk to the dict expected by chunk_repository.insert_chunks."""
    metadata = dict(chunk.metadata) if hasattr(chunk, "metadata") else {}
    metadata.setdefault("source_type", source_type)
    return {
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "title": chunk.title,
        "section_path": " > ".join(chunk.section_path) if chunk.section_path else chunk.title,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_strategy": chunk.chunk_strategy,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }
