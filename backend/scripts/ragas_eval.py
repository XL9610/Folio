#!/usr/bin/env python3
"""Reference-free answer-quality eval via RAGAS + an OpenAI-compatible judge LLM.

Runs each question through the running backend's ``POST /api/ask`` endpoint
(with ``debug_retrieval=true``), collects the ``(question, answer, retrieved
contexts)`` triples, then scores them with RAGAS. No golden corpus and no
ground-truth answers are required for the default metric set - you just need a
judge LLM and a backend that has already ingested documents.

Documents are referenced by a human-readable ``document`` label (e.g.
``"Grokking Algorithms"``) rather than a hard-coded UUID. At startup the script
lists ``GET /api/documents`` and resolves each label to a ``document_id`` by
normalized (alphanumeric-only, case-insensitive) substring match against the
ingested ``filename``/``display_title``. Questions whose label matches zero or
more than one document are reported and skipped, so re-ingesting the corpus (or
moving it to a fresh machine) never invalidates the question set.

Judge LLM: defaults to DeepSeek. Any OpenAI-compatible endpoint works.

Install (dedicated venv, alongside the app deps):
    pip install -r requirements.txt -r requirements-eval.txt

Usage:
    python backend/scripts/ragas_eval.py --questions backend/scripts/ragas_questions.json --out ragas_report.json
    python backend/scripts/ragas_eval.py --questions eval_questions.json            # api defaults to localhost:8000

Questions file - a JSON array. ``document`` is a filename label; ``reference`` is
optional and enables the ground-truth metrics:
    [
      {"question": "What does Chapter 3 cover?", "document": "Grokking Algorithms"},
      {"question": "Explain binary search.", "document": "Grokking Algorithms", "reference": "..."},
      {"question": "What is the main topic?", "document": "Economics", "difficulty": "easy"},
      {"question": "Who won the 2026 World Cup?", "expect_refusal": true}
    ]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# backend/ = parent of this script's dir (backend/scripts/)
_BACKEND_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env best-effort so the app's LLM key is visible to the judge.
try:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND_DIR / ".env")
except Exception:  # pragma: no cover - python-dotenv is optional here
    pass

import warnings  # noqa: E402

import httpx  # noqa: E402  (openai depends on httpx, so it is available)
from openai import OpenAI  # noqa: E402

from langchain_huggingface import HuggingFaceEmbeddings as _LCHFEmbeddings  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402

# Local embedding model for answer_relevancy's question <-> answer similarity.
# Matches the app's default embedder (bge-small-en-v1.5); must be cached locally.
EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"

# Flat `ragas.metrics` (deprecated in 0.4.x but still the only metric base that
# `evaluate()` accepts — the `collections` metrics subclass `BaseMetric`, which
# `evaluate()` rejects). Judge LLM is injected via `evaluate(llm=...)`, embeddings
# via `evaluate(embeddings=...)`. Pinned to ragas==0.4.3.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (  # noqa: E402
        AnswerRelevancy,
        ContextRecall,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextPrecisionWithReference,
    )

DEFAULT_QUESTIONS = [
    {"question": "What is the main topic of the selected source?", "document": None},
    {"question": "What are the key concepts introduced in this source?", "document": None},
    {"question": "Summarize the first chapter.", "document": None},
]


def _norm(text: str) -> str:
    """Alphanumeric-only lowercase key, so 'A History of Western Music' and
    'A_History_of_Western_Music' compare equal and labels survive re-uploads."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def list_documents(api: str, notebook_id: str) -> list[dict]:
    """Fetch the notebook's documents from GET /api/documents."""
    with httpx.Client(timeout=60) as client:
        resp = client.get(f"{api}/api/documents", params={"notebook_id": notebook_id})
        resp.raise_for_status()
        return resp.json()


def resolve_documents(documents: list[dict], questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Resolve each question's `document` label to a `document_id`.

    Returns (resolved_questions, unresolved) where unresolved is a list of
    {question, document, reason} for labels matching zero or >1 documents.
    """
    resolved: list[dict] = []
    unresolved: list[dict] = []

    for entry in questions:
        label = (entry.get("document") or "").strip()
        if not label:
            # No document scope: ask across the whole notebook.
            resolved.append(entry)
            continue

        key = _norm(label)
        matches = [
            d for d in documents
            if key in _norm(d.get("filename", "")) or key in _norm(d.get("display_title", ""))
        ]
        if len(matches) == 1:
            out = dict(entry)
            out["document_ids"] = [matches[0]["document_id"]]
            resolved.append(out)
        elif not matches:
            unresolved.append({"question": entry.get("question"), "document": label, "reason": "no matching document"})
        else:
            names = ", ".join(d.get("filename", "?") for d in matches)
            unresolved.append({"question": entry.get("question"), "document": label, "reason": f"ambiguous: {names}"})

    return resolved, unresolved


def ask_backend(api: str, question: str, notebook_id: str, document_ids: list[str] | None, top_k: int) -> dict:
    """POST one question to /api/ask and return the JSON response."""
    payload: dict = {
        "notebook_id": notebook_id,
        "query": question,
        "top_k": top_k,
        "history": [],
        "allow_web": False,
        "debug_retrieval": True,
    }
    if document_ids:
        payload["document_ids"] = document_ids
    with httpx.Client(timeout=180) as client:
        resp = client.post(f"{api}/api/ask", json=payload)
        resp.raise_for_status()
        return resp.json()


def resolve_judge(args: argparse.Namespace) -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the judge LLM, CLI > env > default."""
    api_key = (
        args.judge_api_key
        or os.environ.get("RAGAS_JUDGE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )
    base_url = (
        args.judge_base_url
        or os.environ.get("RAGAS_JUDGE_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model = (
        args.judge_model
        or os.environ.get("RAGAS_JUDGE_MODEL")
        or os.environ.get("LLM_MODEL")
        or "deepseek-chat"
    )
    if not api_key:
        sys.exit(
            "No judge API key found. Set RAGAS_JUDGE_API_KEY / LLM_API_KEY / DEEPSEEK_API_KEY, "
            "or pass --judge-api-key."
        )
    return base_url, api_key, model


_REFUSAL_PHRASES = (
    "does not cover",
    "doesn't cover",
    "does not contain enough information",
    "don't have any material",
    "no material on",
    "cannot be identified from the source",
    "no reliable",
    "does not discuss",
    "doesn't discuss",
    "does not mention",
    "not covered",
    "out of scope",
)


def _classify_skip(agent_status: str, answer: str) -> str | None:
    """Return a skip reason for a refusal/clarification, else None (scoreable)."""
    if agent_status == "insufficient_context":
        return "refused (insufficient_context)"
    if agent_status == "clarification_needed":
        return "clarification_needed"
    if agent_status != "completed":
        return f"non-completed: {agent_status}"
    if any(p in answer.lower() for p in _REFUSAL_PHRASES):
        return "refused (refusal wording in answer)"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="Backend base URL (default %(default)s)")
    parser.add_argument("--questions", type=Path, help="Path to a JSON array of question objects")
    parser.add_argument("--notebook-id", default="default", help="Notebook to query (default %(default)s)")
    parser.add_argument("--top-k", type=int, default=10, help="Chunks to retrieve per question (default %(default)s)")
    parser.add_argument("--out", type=Path, help="Write the JSON report to this file instead of stdout")
    parser.add_argument("--judge-base-url", help="Judge LLM base URL (default DeepSeek)")
    parser.add_argument("--judge-api-key", help="Judge LLM API key")
    parser.add_argument("--judge-model", help="Judge LLM model name (default deepseek-chat)")
    args = parser.parse_args()

    # ── Load questions ──
    if args.questions:
        questions = json.loads(args.questions.read_text(encoding="utf-8"))
    else:
        print(f"No --questions given; using {len(DEFAULT_QUESTIONS)} built-in placeholder questions.")
        questions = DEFAULT_QUESTIONS
    if not isinstance(questions, list) or not questions:
        sys.exit("--questions must point to a non-empty JSON array.")

    # ── Resolve document labels -> document_ids ──
    documents = list_documents(args.api, args.notebook_id)
    questions, unresolved = resolve_documents(documents, questions)
    if not questions:
        sys.exit(f"No questions could be resolved against the notebook ({len(unresolved)} unresolved).")

    # ── Collect (question, answer, contexts) from the backend ──
    records: list[dict] = []
    skipped: list[dict] = []
    meta: dict[str, dict] = {}
    for entry in questions:
        question = entry.get("question") or entry.get("query")
        if not question:
            skipped.append({"reason": "missing 'question'", "entry": entry})
            continue
        meta[question] = {
            "difficulty": entry.get("difficulty", ""),
            "expect_refusal": bool(entry.get("expect_refusal")),
            "document": entry.get("document", ""),
        }
        try:
            resp = ask_backend(
                args.api,
                question,
                args.notebook_id,
                entry.get("document_ids"),
                args.top_k,
            )
        except Exception as exc:  # network / backend error → skip, keep going
            skipped.append({"question": question, "reason": f"ask failed: {exc}"})
            continue

        answer = (resp.get("answer") or "").strip()
        contexts = [c.get("text", "").strip() for c in resp.get("retrieved_chunks", [])]
        contexts = [c for c in contexts if c]
        agent_status = resp.get("agent_status", "completed")

        skip_reason = _classify_skip(agent_status, answer)
        if skip_reason:
            skipped.append({
                "question": question,
                "agent_status": agent_status,
                "difficulty": meta[question]["difficulty"],
                "expect_refusal": meta[question]["expect_refusal"],
                "document": meta[question]["document"],
                "reason": skip_reason,
            })
            continue

        record = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        reference = entry.get("reference")
        if reference:
            record["reference"] = reference

        if not answer or not contexts:
            skipped.append({
                "question": question,
                "agent_status": agent_status,
                "difficulty": meta[question]["difficulty"],
                "expect_refusal": meta[question]["expect_refusal"],
                "document": meta[question]["document"],
                "reason": "no answer or no contexts",
            })
            continue
        records.append(record)

    if not records:
        sys.exit(f"Nothing scoreable: {len(skipped)} skipped, {len(unresolved)} unresolved (see report).")

    # ── Judge LLM + embeddings ──
    base_url, api_key, model = resolve_judge(args)
    judge = llm_factory(model, client=OpenAI(base_url=base_url, api_key=api_key), max_tokens=8192)
    embeddings = LangchainEmbeddingsWrapper(_LCHFEmbeddings(model_name=EMBEDDING_MODEL))

    # ── Score: reference-free metrics over every sample ──
    ref_free_metrics = [Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithoutReference()]
    ref_free_records = [
        {k: v for k, v in r.items() if k != "reference"} for r in records
    ]
    result = evaluate(
        dataset=EvaluationDataset.from_list(ref_free_records),
        metrics=ref_free_metrics,
        llm=judge,
        embeddings=embeddings,
    )
    rows = result.to_pandas().to_dict(orient="records")
    metric_cols = [m.name for m in ref_free_metrics]

    # ── Ground-truth metrics over the subset that supplied a reference ──
    ref_records = [r for r in records if r.get("reference")]
    if ref_records:
        ref_metrics = [ContextRecall(), LLMContextPrecisionWithReference()]
        ref_result = evaluate(
            dataset=EvaluationDataset.from_list(ref_records),
            metrics=ref_metrics,
            llm=judge,
            embeddings=embeddings,
        )
        metric_cols += [m.name for m in ref_metrics]
        ref_by_question = {r["user_input"]: r for r in ref_result.to_pandas().to_dict(orient="records")}
        for row in rows:
            extra = ref_by_question.get(row.get("user_input"))
            if extra:
                row.update({k: v for k, v in extra.items() if k not in ("user_input", "retrieved_contexts", "response")})

    means: dict = {}
    for name in metric_cols:
        vals = [r.get(name) for r in rows]
        vals = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        means[name] = round(sum(vals) / len(vals), 4) if vals else None

    for row in rows:
        m = meta.get(row.get("user_input"))
        if m:
            row["difficulty"] = m["difficulty"]
            row["expect_refusal"] = m["expect_refusal"]
            row["document"] = m["document"]

    report = {
        "judge": {"base_url": base_url, "model": model},
        "scored": len(records),
        "skipped": skipped,
        "unresolved": unresolved,
        "metric_means": means,
        "samples": rows,
    }

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
