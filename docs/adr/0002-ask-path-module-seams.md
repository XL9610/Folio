# 2. Ask-path module seams — conservative helper extraction

- **Status:** Accepted
- **Date:** 2026-09-21

## Context

The ask path lived in two of the repo's largest non-test modules:
`backend/app/routers/ask_router.py` (~1246 lines) and
`backend/app/services/qa/tools.py` (~1107 lines). Neither was tangled
internally so much as *mixed*: `ask_router.py` held the FastAPI contract, the
`AgenticAskPipeline` class, deterministic structural Q&A, and the eval/debug
export surface in one file; `tools.py` held the `QAToolExecutor` alongside
general-purpose section-label text utilities that `agent_generator.py` reached
into the executor's module to import.

The audit's goal was to split each file by cohesion without changing behavior.
The decisive constraint came from `unittest.mock.patch`, which resolves
attributes on the **defining** module's globals, not on re-export facades:

- **34 sites** patch `app.routers.ask_router.{OpenAICompatibleClient,
  HybridRetriever, SentenceTransformerEmbedder, QdrantStore, LLMConfig,
  QAToolExecutor}`.
- **13 sites** patch `app.services.qa.tools.{get_document,
  get_document_sections, get_chunks_by_section_prefix,
  get_chunks_for_section_paths}`.
- **2 sites** patch the `get_chunk_by_id` / `get_document` names that the
  eval/debug export functions read.

A re-export facade keeps *imports* working but not *patches*. So any symbol
with a patch site had to stay where it was; only zero-patch-site symbols were
safe to move mechanically.

## Decision

Move only cohesive symbol clusters with **zero patch sites**, one commit per
layer, keeping bodies verbatim:

- **`ask_router.py`** → `routers/ask_schemas.py` (Pydantic models),
  `services/qa/follow_up.py` (follow-up query expansion),
  `services/qa/structural_answers.py` (deterministic chapter/title/page
  answers), `services/qa/source_framing.py` (source-framing question
  enrichment), `services/qa/ask_debug.py` (eval/debug exports). The router,
  `MAX_AGENT_ROUNDS`, `MAX_WALL_CLOCK_SECONDS`, `_agent`, `AgenticAskPipeline`,
  `_merge_tool_results`, `_get_agent`, and the endpoint stay. `_agent` and
  `_get_agent` must stay because `config_router.py` writes `ask_router._agent`
  through a bare `try/except`; a moved singleton would fail silently.
- **`tools.py`** → `services/qa/section_labels.py` (section-label text
  utilities) and `services/qa/citations.py` (per-round `_finalize_chunk_citations`
  plus post-merge `finalize_merged_citations`, uniting all citation shaping
  from the Step 3.1 work). `SectionMatch`, `ToolResults`, `ToolFn`, the rerank
  gate, `QAToolExecutor`, and all six handlers stay. Three thin
  `@staticmethod` delegators on `QAToolExecutor` forward to `citations.py` so
  the private API tests call (`executor._finalize_chunk_citations`) keeps
  working; `citations.py` imports `ToolResults` only under `TYPE_CHECKING`, so
  `tools.py` → `citations.py` has no runtime cycle.

Both extraction commits append a re-export block so the historical import
surface (`from app.routers.ask_router import ...`, `from app.services.qa.tools
import ...`) keeps resolving. The one test repoint was
`test_eval_debug_retrieval.py`, whose two `monkeypatch.setattr` calls now target
`app.services.qa.ask_debug` (the module that actually reads the names).

## Consequences

- `ask_router.py` and `tools.py` now read as a contract + pipeline and an
  executor, respectively; structural Q&A, citation shaping, and debug export
  are each one hop away.
- The split stops at the patch-target boundary. The six tool handlers and the
  `execute()` prelude read repository functions that 13 test sites patch by
  position, so moving them would turn a mechanical refactor into a 13-site
  repointing exercise with no behavioral gain. Revisit only if `tools.py`
  grows a seventh tool.
- `tools.py` is deliberately **not** a pure re-export facade, and the handlers
  are deliberately **not** a mixin package: a facade would break the patches,
  and a mixin used by exactly one class is scatter, not cohesion.
- No behavior change: every moved function kept its body verbatim; the only
  relocated-code edit was the `QAToolExecutor._dedupe_chunks` → `_dedupe_chunks`
  self-call inside `_finalize_chunk_citations`, a mechanical consequence of a
  method becoming a module function.
