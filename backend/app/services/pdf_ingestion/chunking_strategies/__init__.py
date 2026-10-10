"""Chunking strategy modules.

Each strategy follows the signature:
    (pages: list[DocumentPage], doc_id: str, **kwargs) -> ChunkingResult
"""

from __future__ import annotations
