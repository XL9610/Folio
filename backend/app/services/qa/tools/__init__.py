"""QAToolExecutor — executes white-listed tools for the agentic QA pipeline.

Tools (registry):
- get_source_metadata: Returns document filename, page count, chunk count.
- get_source_outline: Returns section structure (section_path, page_start, page_end).
- retrieve_chunks: Hybrid dense + FTS retrieval with reranking.
- find_sections: Locate sections matching a natural-language hint.
- get_section_chunks: Retrieve chunks belonging to specific sections.

All tools enforce document_ids boundary and notebook scope.

Package layout: executor.py (QAToolExecutor), types.py (ToolResults,
SectionMatch), and one module per tool group: sources.py, retrieve.py,
sections.py, web.py. Names below stay importable from `qa.tools`.
"""

from ..agent_planner import ToolCall  # noqa: F401
from ..citations import (  # noqa: F401
    MAX_BOOK_CITATIONS,
    MAX_WEB_CITATIONS,
    finalize_merged_citations,
)
from ..section_labels import (  # noqa: F401
    extract_numbered_section_label,
    extract_numbered_section_prefix,
    is_page_window_section_label,
    resolve_chapter_page_range,
    section_label_for_display,
)
from .executor import QAToolExecutor  # noqa: F401
from .retrieve import _should_rerank  # noqa: F401
from .types import SectionMatch, ToolResults  # noqa: F401
