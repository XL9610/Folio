# Folio RAGAS Evaluation Report

Evaluation date: 2026-10-10 (Asia/Tokyo)

## Setup

- System under test: Folio local-first RAG application.
- Answer model: DeepSeek `deepseek-chat` via `https://api.deepseek.com/v1`.
- Judge model: OpenAI `gpt-4o` via `https://api.openai.com/v1`.
- RAGAS version: `0.4.3`.
- Corpus: all 21 non-`.gitkeep` files under `books/`, ingested into isolated notebook `b7b1a9bc-ee76-4f44-831b-5b7652bd9e7a`.
- Final ingestion state: 21 `completed`, 0 `fts_only`, 0 `indexed_partial`, 0 `failed`; all 21 had dense indexes.
- Question set: all 61 entries in `backend/scripts/ragas_questions.json`, unchanged and run in their original order.
- Final disposition: 53 scored, 8 skipped, 0 unresolved. Six skipped samples were expected refusal tests; two were unexpected refusals.
- Retrieval and evaluation settings were unchanged (`top_k=10`, existing prompts and metrics, judge `max_tokens=8192`).

The pre-existing `default` notebook contained duplicate historical ingestions, making 51 labels ambiguous under the runner's unchanged matching rule. To avoid deleting user data or changing the resolver, the corpus was freshly ingested once into the isolated notebook above. A sequential ingestion pass was used after a concurrent attempt exposed SQLite write-lock failures.

## Overall means

The runner's `metric_means` object, verbatim:

```json
{
  "faithfulness": 0.9676,
  "answer_relevancy": 0.8592,
  "llm_context_precision_without_reference": 0.9032,
  "context_recall": 1.0,
  "llm_context_precision_with_reference": 0.9306
}
```

The two reference-based metrics cover only the four questions that supplied a `reference`; all four were easy questions. Their perfect recall should not be generalized to the full set.

## Breakdown by difficulty

Means exclude skipped samples and absent reference-only metrics. `Scored / total` exposes the two unexpected refusals.

| Difficulty | Scored / total | Faithfulness | Answer relevancy | Context precision (no reference) | Context recall | Context precision (with reference) |
|---|---:|---:|---:|---:|---:|---:|
| Easy | 23 / 24 | 0.9617 | 0.8344 | 0.8983 | 1.0000 | 0.9306 |
| Medium | 23 / 24 | 0.9697 | 0.8779 | 0.9275 | — | — |
| Hard | 7 / 7 | 0.9801 | 0.8794 | 0.8398 | — | — |

Hard questions did not show a broad score collapse: faithfulness and answer relevancy were slightly higher than for easy questions. Retrieval precision did weaken on hard questions, falling to 0.8398 from 0.8983 on easy and 0.9275 on medium; the hard presentation-summary sample was especially weak at 0.4186.

## Refusal check

All 6/6 `expect_refusal=true` questions were behaviorally refused, for a 100% behavioral refusal rate. This was checked against fresh answer text after the scored run, not merely against the runner's `skipped` classification. The recheck used the same isolated notebook, document scope, and `top_k=10`.

| Question | Agent status | Answer-text sanity check | Result |
|---|---|---|---|
| Who won the 2026 FIFA World Cup? | `insufficient_context` | Said the selected sources lacked enough information. | Pass |
| What is the capital of France? | `insufficient_context` | Said the selected sources lacked enough information. | Pass |
| What is the current price of Bitcoin? | `clarification_needed` | Explained that live market data was unavailable with web research disabled and asked whether to enable it. | Pass |
| Summarize the plot of the movie Inception. | `completed` | Explicitly said the retrieved material did not cover the movie and declined to invent a summary. | Pass |
| Explain how neural networks are trained, according to this book on Western music. | `completed` | Explicitly said the source did not cover neural networks; related music-training material was labeled as an analogy. | Pass |
| What does Grokking Algorithms say about the history of opera? | `completed` | Explicitly said the book did not cover opera and grounded that refusal in its algorithmic contents. | Pass |

There were also two false refusals among the 55 answerable questions (3.6%): one easy and one medium question.

## Failures and anomalies

### Final-run scoring failures

- `NaN` metrics: none.
- Unresolved documents: none; every document-scoped label resolved uniquely, and all 61 questions entered the run.
- Final ingestion failures: none; all 21 documents completed with dense indexes.
- Question-time technical crashes: none; all 61 formal `/api/ask` requests returned HTTP 200.
- Unexpectedly skipped non-refusal questions:
  - “What does the author mean by being fooled by randomness?” — `agent_status=completed`, but the answer contained refusal wording and was skipped.
  - “What role do the Aes Sedai play in the story?” — `agent_status=insufficient_context`, skipped as a refusal.

### Material low-score cases

- “What is the main news story covered in this magazine issue?” had faithfulness 0.2857, by far the weakest grounding score.
- “How does the frame narrative structure of Wuthering Heights work?” had answer relevancy 0.0000 and context precision 0.5437.
- “Qu'est-ce que la philosophie ?” had answer relevancy 0.0000 despite faithfulness 0.9783 and context precision 0.9815. This mismatch may reflect answer-relevancy judge or multilingual sensitivity, but the score remains a failure signal.
- “Who is the central protagonist of The Fires of Heaven?” had context precision 0.0000 despite faithfulness 1.0000.
- “Summarize the key concepts covered across the slides of this module.” had answer relevancy 0.6346 and context precision 0.4186.

### Technical incidents during execution

These incidents occurred before or around the valid final run and were not silently folded into answer quality:

- Two initial MHTML/ODT ingestion jobs failed because the dedicated environment lacked the repository's optional `kreuzberg` extractor dependency. Installing `backend/requirements-extractors.txt` and retrying resolved both.
- An early MHTML retry became `indexed_partial` after a local embedding error (`Cannot copy out of meta tensor`). It was not used in the final evaluation; clean sequential ingestion completed the same file with a dense index.
- Bulk ingestion into the first isolated notebook caused SQLite `database is locked` errors, including one HTTP 500 from the ingest endpoint and status-update failures. That evaluation-only notebook was verified, deleted, and rebuilt from `books/` one file at a time; the final notebook has no failed or partial documents.
- The configured `RAGAS_JUDGE_API_KEY` returned HTTP 401 `expired_secret_key` in an invalid preliminary scoring attempt. As allowed by the task prompt, the formal run temporarily used the `.env` `OPENAI_API_KEY` while keeping the judge endpoint and model at OpenAI GPT-4o.
- The backend stored the `ë` in the Wuthering Heights multipart filename as mojibake. This broke an exact-filename polling check, but the runner's normalized label still resolved uniquely and the document completed successfully.
- The refusal-text audit's first PowerShell filter mistakenly began re-asking the full array instead of only refusal items. It was interrupted after roughly 20 extra read-only calls; those calls did not alter the corpus or formal JSON report. The corrected audit produced the six findings above.
- RAGAS repeatedly warned `LLM returned 1 generations instead of requested 3` for answer relevancy. This is the prompt's known benign warning; no parameters were changed to suppress it.

## Complete per-sample results

`CP-no-ref` is `llm_context_precision_without_reference`; `CP-ref` is `llm_context_precision_with_reference`. An em dash means the metric was inapplicable or the sample was skipped, not `NaN`.

| # | Difficulty | Document scope | Question | Disposition | Faithfulness | Answer relevancy | CP-no-ref | Context recall | CP-ref |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | easy | Grokking Algorithms | What is binary search and why is it faster than simple search? | scored | 1.0000 | 0.9718 | 1.0000 | 1.0000 | 1.0000 |
| 2 | easy | Grokking Algorithms | What does Big O notation describe? | scored | 1.0000 | 0.9115 | 0.9627 | — | — |
| 3 | medium | Grokking Algorithms | What is a hash table and when would you use one? | scored | 1.0000 | 0.9495 | 0.8866 | — | — |
| 4 | medium | Grokking Algorithms | What is recursion, and what are its base case and recursive case? | scored | 1.0000 | 0.8124 | 1.0000 | — | — |
| 5 | hard | Grokking Algorithms | Compare breadth-first search and Dijkstra's algorithm. | scored | 1.0000 | 0.8640 | 0.8211 | — | — |
| 6 | easy | Data Structures and Algorithms in Java | What is the difference between an array and a linked list? | scored | 0.9474 | 0.9221 | 1.0000 | 1.0000 | 0.8056 |
| 7 | easy | Data Structures and Algorithms in Java | What is a stack and which operations does it support? | scored | 1.0000 | 0.7659 | 1.0000 | — | — |
| 8 | medium | Data Structures and Algorithms in Java | What is a binary search tree and what property distinguishes it? | scored | 1.0000 | 0.9525 | 1.0000 | — | — |
| 9 | hard | Data Structures and Algorithms in Java | Compare merge sort and quick sort in terms of time complexity and tradeoffs. | scored | 0.9412 | 0.9497 | 0.9167 | — | — |
| 10 | easy | Think Python | What is a function in Python? | scored | 1.0000 | 1.0000 | 0.9821 | — | — |
| 11 | easy | Think Python | What is a list in Python and how does it differ from a tuple? | scored | 1.0000 | 0.9515 | 0.7381 | — | — |
| 12 | medium | Think Python | Explain recursion using the factorial example. | scored | 1.0000 | 0.9385 | 1.0000 | — | — |
| 13 | medium | Think Python | What is a class and what is an object in object-oriented Python? | scored | 1.0000 | 0.6932 | 0.7875 | — | — |
| 14 | easy | Economics | What is opportunity cost? | scored | 1.0000 | 0.8656 | 0.9861 | 1.0000 | 1.0000 |
| 15 | easy | Economics | What is supply and demand? | scored | 1.0000 | 0.5626 | 0.9667 | — | — |
| 16 | medium | Economics | What is inflation and what causes it? | scored | 1.0000 | 1.0000 | 0.9167 | — | — |
| 17 | medium | Economics | Explain the difference between microeconomics and macroeconomics. | scored | 0.9737 | 0.9451 | 1.0000 | — | — |
| 18 | easy | AI Engineering | What is a foundation model? | scored | 1.0000 | 0.8261 | 0.8909 | — | — |
| 19 | easy | AI Engineering | What is retrieval-augmented generation (RAG) and why is it useful? | scored | 1.0000 | 0.9641 | 1.0000 | 1.0000 | 0.9167 |
| 20 | medium | AI Engineering | Explain the difference between fine-tuning and prompt engineering. | scored | 1.0000 | 0.9809 | 0.9821 | — | — |
| 21 | hard | AI Engineering | What are the tradeoffs between RAG and fine-tuning an LLM? | scored | 1.0000 | 0.8924 | 1.0000 | — | — |
| 22 | easy | History of Western Music | What is Gregorian chant? | scored | 1.0000 | 0.8682 | 0.9617 | — | — |
| 23 | medium | History of Western Music | What is a symphony? | scored | 1.0000 | 0.8509 | 1.0000 | — | — |
| 24 | hard | History of Western Music | Explain the differences between the Renaissance and Baroque periods in music. | scored | 1.0000 | 0.9013 | 0.8056 | — | — |
| 25 | medium | Art of Art History | Who were Vasari and Winckelmann, and what did each contribute? | scored | 0.8571 | 0.8487 | 0.8333 | — | — |
| 26 | medium | Art of Art History | Explain the idea of the 'art museum as ritual'. | scored | 0.8800 | 0.8183 | 1.0000 | — | — |
| 27 | medium | Art of Art History | What is the difference between art history and art criticism? | scored | 0.7778 | 0.9268 | 0.7750 | — | — |
| 28 | easy | Vocal Technique | What is breath support in singing? | scored | 0.9583 | 0.8944 | 1.0000 | — | — |
| 29 | medium | Vocal Technique | What is a vocal register? | scored | 1.0000 | 0.9598 | 1.0000 | — | — |
| 30 | hard | Vocal Technique | Explain how resonance shapes vocal tone. | scored | 1.0000 | 0.9332 | 1.0000 | — | — |
| 31 | easy | A3 Thinking | What is A3 thinking? | scored | 1.0000 | 0.9031 | 1.0000 | — | — |
| 32 | medium | A3 Thinking | What are the steps of the A3 problem-solving process? | scored | 1.0000 | 1.0000 | 0.7436 | — | — |
| 33 | easy | Trading Stocks | What is value investing? | scored | 1.0000 | 0.8545 | 0.9429 | — | — |
| 34 | medium | Trading Stocks | What is a competitive 'moat' in investing? | scored | 0.9677 | 0.9291 | 0.8875 | — | — |
| 35 | medium | Trading Stocks | What is the margin of safety in value investing? | scored | 1.0000 | 1.0000 | 1.0000 | — | — |
| 36 | easy | Immigration law 2011 | What is a student visa in the context of UK immigration law? | scored | 1.0000 | 0.9594 | 0.5317 | — | — |
| 37 | medium | Immigration law 2011 | What are the grounds for deportation under UK immigration law? | scored | 1.0000 | 1.0000 | 1.0000 | — | — |
| 38 | hard | Immigration law 2011 | How does the asylum application process work in the UK? | scored | 0.9750 | 0.9802 | 0.9167 | — | — |
| 39 | easy | US immigration | How did terrorism concerns change US immigration law? | scored | 0.9487 | 0.9348 | 1.0000 | — | — |
| 40 | medium | US immigration | How did the USA PATRIOT Act affect immigration? | scored | 0.9259 | 0.9082 | 1.0000 | — | — |
| 41 | easy | Fooled by Randomness | What does the author mean by being fooled by randomness? | skipped: refused (refusal wording in answer) | — | — | — | — | — |
| 42 | medium | Fooled by Randomness | What is survivorship bias? | scored | 1.0000 | 0.9181 | 0.9809 | — | — |
| 43 | medium | Fooled by Randomness | Explain the role of luck versus skill in trading success, according to the author. | scored | 1.0000 | 0.9434 | 1.0000 | — | — |
| 44 | easy | Wuthering Heights | Who is Heathcliff in Wuthering Heights? | scored | 1.0000 | 0.7619 | 0.9386 | — | — |
| 45 | easy | Wuthering Heights | Who narrates Wuthering Heights? | scored | 1.0000 | 0.8244 | 1.0000 | — | — |
| 46 | medium | Wuthering Heights | How does the frame narrative structure of Wuthering Heights work? | scored | 0.9500 | 0.0000 | 0.5437 | — | — |
| 47 | easy | Fires of Heaven | Who is the central protagonist of The Fires of Heaven? | scored | 1.0000 | 0.8630 | 0.0000 | — | — |
| 48 | medium | Fires of Heaven | What role do the Aes Sedai play in the story? | skipped: refused (insufficient_context) | — | — | — | — | — |
| 49 | easy | 5940 Mod6 | What is the main topic of Module 6 in this presentation? | scored | 1.0000 | 0.8499 | 0.7968 | — | — |
| 50 | hard | 5940 Mod6 | Summarize the key concepts covered across the slides of this module. | scored | 0.9444 | 0.6346 | 0.4186 | — | — |
| 51 | easy | 2605.06647 | What is a Superintelligent Retrieval Agent (SIRA)? | scored | 1.0000 | 0.9871 | 0.9821 | — | — |
| 52 | medium | 2605.06647 | What problem does SIRA aim to solve? | scored | 1.0000 | 1.0000 | 1.0000 | — | — |
| 53 | easy | Philosophie | Qu'est-ce que la philosophie ? | scored | 0.9783 | 0.0000 | 0.9815 | — | — |
| 54 | medium | Philosophie | Quelle est la difference entre la morale et l'ethique ? | scored | 0.9706 | 0.8170 | 0.9946 | — | — |
| 55 | easy | US News | What is the main news story covered in this magazine issue? | scored | 0.2857 | 0.7502 | 1.0000 | — | — |
| 56 | unanswerable | all sources | Who won the 2026 FIFA World Cup? | skipped: refused (insufficient_context) | — | — | — | — | — |
| 57 | unanswerable | all sources | What is the capital of France? | skipped: refused (insufficient_context) | — | — | — | — | — |
| 58 | unanswerable | all sources | What is the current price of Bitcoin? | skipped: clarification_needed | — | — | — | — | — |
| 59 | unanswerable | all sources | Summarize the plot of the movie Inception. | skipped: refused (refusal wording in answer) | — | — | — | — | — |
| 60 | unanswerable | History of Western Music | Explain how neural networks are trained, according to this book on Western music. | skipped: refused (refusal wording in answer) | — | — | — | — | — |
| 61 | unanswerable | Grokking Algorithms | What does Grokking Algorithms say about the history of opera? | skipped: refused (refusal wording in answer) | — | — | — | — | — |

## Bottom line

Folio was usually well grounded and retrieved relevant evidence, with overall faithfulness 0.9676 and context precision 0.9032, and it correctly refused all six deliberately unanswerable prompts. Its main weaknesses are uneven retrieval on particular books and cross-document summaries, two false refusals on answerable questions, and severe sample-level failures that the high averages conceal. Hard questions did not collapse overall, but their lower context precision indicates that broader synthesis is less selective. Finally, GPT-4o is an LLM judge rather than a human evaluator and may be lenient or inconsistent—especially given the one-generation answer-relevancy warning and the small four-question reference subset.
