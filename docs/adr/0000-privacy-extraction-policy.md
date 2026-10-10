# ADR: Privacy Extraction Policy

**Status:** Accepted
**Date:** 2026-07-02

## Decision

All document extraction and embedding runs **locally by default**.
No document text or extracted content leaves the user's machine
unless the user explicitly configures cloud services.

## Tier 1 — Local-Only (Default)

Activated when no cloud API keys are configured.

| Component | Engine | Network Required |
|-----------|--------|------------------|
| PDF extraction | PyMuPDF (fitz) | No |
| EPUB extraction | ebooklib | No |
| PPTX extraction | python-pptx | No |
| DOCX extraction | python-docx | No |
| URL extraction | trafilatura | No (fetches URL content, but does not send to third-party API) |
| Fallback extraction | Kreuzberg | No |
| Embedding | sentence-transformers (bge-small-en-v1.5, local model) | No |
| LLM generation | None (Ask API returns 503 without LLM config) | N/A |

## Tier 2 — Cloud Opt-In

Each cloud integration requires a user-provided API key.

| Component | Provider | Requires |
|-----------|----------|----------|
| Cloud embedding | OpenAI text-embedding-3-small | `OPENAI_API_KEY` |
| Cloud LLM | Anthropic Claude / OpenAI GPT | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` |

**No cloud extraction is offered.** All extraction formats use local libraries.

## Rationale

- **Default privacy**: Users' textbooks and study materials are often copyrighted.
  Defaulting to local-only prevents accidental upload to cloud services.
- **Offline capability**: Students may use the app without internet access.
- **Kreuzberg is local**: Unlike some alternatives, Kreuzberg processes files
  entirely on-device with no external API calls.
- **User control**: Cloud features must be opt-in with clear labeling.

## Consequences

- default embedding model (bge-small) must be downloaded on first run (~130 MB)
- cloud embedding requires rebuild of the dense vector index (dimension mismatch)
- `eval runner` must support retrieval-only mode (does not require LLM)
