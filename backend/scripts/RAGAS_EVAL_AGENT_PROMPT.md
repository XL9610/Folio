# Task: Run a RAG evaluation of the Folio study-companion app

You are running an evaluation of a local-first RAG question-answering app called
**Folio**. Your job is to run the whole eval pipeline end to end — **ingest the
corpus, score the answers, and produce an honest, complete report**. You are the
evaluator, not a promoter: report every weakness, not just the good numbers.

## What you are testing

- The app ingests textbooks/papers into a notebook and answers questions with
  citations. Its answer-generation LLM is **DeepSeek** (`LLM_API_KEY` in
  `backend/.env`) — this is the model under test.
- The eval scores each `(question, answer, retrieved-contexts)` triple with
  RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, and —
  where a question carries a `reference` — `context_recall` and
  `context_precision_with_reference`.

## The corpus (files you will ingest)

The source documents live under `books/`, organized by format
(`books/PDF/`, `books/EPUB/`, `books/DOCX/`, `books/PPTX/`, `books/HTML/`,
`books/samples/`). The question set spans ~20 of these files — algorithms,
Python, economics, AI engineering, music history, art history, vocal technique,
A3 thinking, investing, immigration law, two novels, a French philosophy text,
a presentation deck, an arXiv paper (`2605.06647v1.pdf`, the "Superintelligent
Retrieval Agent" paper, referenced in questions by its ID `2605.06647`), and a
magazine issue. Ingest **every** non-`.gitkeep` file under `books/`.

## Judge model: GPT-4o (OpenAI)

The RAGAS judge is **GPT-4o** via `RAGAS_JUDGE_*` in `backend/.env`
(`https://api.openai.com/v1`). The tested model (answers) is still DeepSeek, so
the judge and the tested model are **different** — there is no self-judging bias.
Still state the usual LLM-judge leniency caveat in the report.

## The fairness rules (do not violate these)

1. **Do not modify the question set.** `backend/scripts/ragas_questions.json`
   has 61 questions. Run all of them, in order, as-is. Do not delete hard
   questions, do not add easy ones, do not reword.
2. **Report everything.** Include every sample: high scores, low scores, `NaN`
   (a metric that failed to compute), `unresolved` (document label that matched
   no ingested file), and `skipped` (refusals / no contexts). Silently dropping
   failures invalidates the eval.
3. **Do not tune to inflate scores.** Do not change the RAGAS metrics, judge
   temperature/max_tokens, the app's retrieval settings, or the prompt
   templates. Run them as they are.
4. **Refusals are data, not errors.** Questions with `"expect_refusal": true`
   are *supposed* to be refused. The runner classifies them by `agent_status`
   into the `skipped` list — `insufficient_context` → `refused
   (insufficient_context)`, `clarification_needed` → `clarification_needed`.
   Count a refusal or clarification as a pass, and cross-check the answer text
   for refusal wording as a sanity check.
5. **Flag any technical crash separately** (e.g. an HTTP 503 or a failed
   ingestion) — it is neither a correct answer nor a correct refusal.

## Files you need

- `backend/scripts/ragas_questions.json` — the question set (61 questions; each
  references a source by a human-readable `document` label, not a UUID).
- `backend/scripts/ragas_eval.py` — the eval runner. It lists
  `GET /api/documents`, resolves each `document` label to a `document_id`, asks
  the backend, scores with RAGAS, and writes a JSON report.
- `backend/requirements-eval.txt` — eval-only dependencies.
- `backend/.env` — contains `LLM_API_KEY` (DeepSeek, tested model). Read it; do
  not commit it.

## Setup

1. Create a dedicated venv and install the app + eval deps:
   ```bash
   cd backend
   python -m venv .ragas-venv
   .ragas-venv/Scripts/activate   # Windows  (source .ragas-venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt -r requirements-eval.txt
   ```
2. **Point the judge at GPT-4o.** In `backend/.env`, ensure `RAGAS_JUDGE_API_KEY`,
   `RAGAS_JUDGE_BASE_URL=https://api.openai.com/v1`, and
   `RAGAS_JUDGE_MODEL=gpt-4o` are set (uncommented). The runner prefers
   `RAGAS_JUDGE_*` over the app's DeepSeek `LLM_*` config. If the GPT judge key
   is out of credit, swap `RAGAS_JUDGE_API_KEY` to the app's `OPENAI_API_KEY`
   value.
3. Start the backend (answers questions, uses DeepSeek):
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
4. Verify: `GET http://127.0.0.1:8000/health` → `{"status":"ok", ...}`.

## Ingest the corpus

The runner resolves document labels against whatever is already ingested, so
**you must ingest the books first**. For every non-`.gitkeep` file under
`books/`, POST it to the ingest endpoint (multipart field name is `file`):

```python
import pathlib, httpx
for p in sorted(pathlib.Path("books").rglob("*")):
    if not p.is_file() or p.name == ".gitkeep":
        continue
    with p.open("rb") as fh:
        r = httpx.post(
            "http://127.0.0.1:8000/api/documents/ingest",
            params={"notebook_id": "default"},
            files={"file": (p.name, fh)},
            timeout=600,
        )
        r.raise_for_status()
        print(p.name, r.json()["status"])
```

Ingestion is **async** — each call returns `{"status": "queued"}` and processes
in the background. Poll until every document reaches a terminal state:

```bash
curl -s "http://127.0.0.1:8000/api/documents?notebook_id=default"
```

All documents must be `completed`, `fts_only`, or `indexed_partial` (not
`pending` / `indexing`) before you run the eval. Note the filenames match the
source files, so the runner's label matching will resolve correctly. If any
document lands in `failed`, report it separately and continue — the runner will
report its questions as `unresolved`/`skipped`.

## Run

```bash
python backend/scripts/ragas_eval.py \
  --questions backend/scripts/ragas_questions.json \
  --out ragas_report.json
```

Several minutes for 61 questions. The script already loads the local
`bge-small-en-v1.5` embeddings and sets `max_tokens=8192` on the judge so
`faithfulness` is not truncated.

## Produce the report

Read `ragas_report.json` and write a markdown report with:

1. **Setup** — tested model, judge model, question count, scored vs skipped vs
   unresolved.
2. **Overall means** — the `metric_means` object verbatim.
3. **Breakdown by difficulty** — mean of each metric per group (easy / medium /
   hard). If hard scores collapse vs easy, say so.
4. **Refusal check** — how many of the `expect_refusal` questions were actually
   refused, checked against the **answer text** (not just the runner's `skipped`
   status). Report a behavioral refusal rate.
5. **Failures** — every `NaN` metric, every unexpectedly-skipped non-refusal
   question, every `unresolved` document, every failed ingestion, and every
   technical crash (HTTP 5xx), with the question text.
6. **Bottom line** — 3–5 sentences: what the app does well, where it is weak,
   and the usual LLM-judge leniency caveat (the judge is a model, not a human).

## Known gotchas (do not rediscover these)

- `requirements-eval.txt` pins `ragas==0.4.3` and `langchain-community<0.4`
  (ragas 0.4.3 imports a module removed in langchain-community 0.4.x). Do not
  upgrade ragas or remove the pin.
- The eval needs `sentence-transformers` + `langchain-huggingface` (in
  `requirements-eval.txt`); `BAAI/bge-small-en-v1.5` is cached locally.
- "LLM returned 1 generations instead of requested 3" is a benign RAGAS warning
  (answer_relevancy asks for 3 generations). Note it; do not fight it.
- Ingestion of a large PDF can take a few minutes and is CPU-bound on the local
  embedding model — poll patiently; do not re-POST a file that is already
  `completed`.
