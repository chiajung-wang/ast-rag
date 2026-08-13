# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# Working Agreement

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project Reference

## Project

`ast-rag` — RAG-over-code for `langchain-core`. Ask natural-language questions about the codebase; get answers with validated, source-linked citations.

## Commands

```bash
make install   # install deps (uv or pip)
make run       # launch Streamlit UI
make index     # (re)build SQLite index from langchain-core source
make eval      # end-to-end eval on the 50-question dev set -> evals/results/
make eval-test # end-to-end eval on the 17-question held-out set
make eval-retrieval  # retriever-only ablation: recall@k, MRR, nDCG@5 (no LLM, ~free)
make check     # run unit tests (pytest tests/ -v)

python ask.py "Where is Runnable defined?"   # CLI agent test
python query.py "Runnable definition"        # raw retrieval test
python verify_embeddings.py                  # check provider vectors match the index
```

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph (Python) — 2 nodes: `retrieve → answer` |
| Provider | **OpenRouter only** — one key (`OPENROUTER_API_KEY`) for chat and embeddings. Config in `provider.py`. |
| LLM | `AGENT_MODEL`, default `anthropic/claude-haiku-4.5` (OpenRouter slugs use dots: `4.5` not `4-5`) |
| Embeddings | `EMBED_MODEL`, default `openai/text-embedding-3-small`. Recorded in the `.db`; a mismatched query raises `EmbedModelMismatch`. |
| Storage | SQLite + `sqlite-vec` (one `.db` file, zero infrastructure) |
| Parser | Python `ast` stdlib — function, class, and method granularity |
| BM25 | `rank-bm25` with camelCase expansion |
| UI | Streamlit |

## Architecture

```
Indexer (one-time) → SQLite+sqlite-vec (.db file)
                              ↑
LangGraph Agent (2 nodes):
  retrieve → answer (model from AGENT_MODEL env var)
       ↓
  Tools: search_corpus / find_symbol / read_file
       ↓
  Streamlit chat UI
```

Entire system runs in one Python process. No services, no Docker, no separate frontend.

## Key Implementation Details

**Chunk granularity**: one chunk per top-level function, top-level class, and method. Methods are sibling chunks (not nested). Method embed text prefixed with `"ClassName.method_name: "` for BM25 and dense retrieval.

**Hybrid Retrieval**: BM25 top-10 + dense top-10 → reciprocal rank fusion → top-5. Measured recall@5 via `make eval-retrieval` — read the held-out column, not the dev one (see the note below):

| config | dev (45 q) | held-out (15 q) |
|---|---|---|
| BM25 only | 42.2% | 73.3% |
| Dense only | 73.3% | 86.7% |
| RRF hybrid | 68.9% | **93.3%** |
| RRF + symbol pre-check | **95.6%** | 93.3% |

Refreshed 2026-08-12 (A5) — dev was stale at 33 questions after task 6.4 added 12 more without re-running the ablation.

BM25 tokenizer expands camelCase + snake_case (`RunnableSequence` → `["runnable", "sequence", "runnablesequence"]`, `invoke_async` → `["invoke", "async", "invoke_async"]`). Implemented in `retrieval/bm25_index.py`, `retrieval/rrf.py`, `retrieval/pipeline.py`.

**Retrieve node logic**: heuristic pre-check extracts CamelCase/snake_case tokens from query, checks symbol name set. Match → `find_symbol` first, then `search_corpus` for remaining slots. No match → `search_corpus` only.

**Tools** (plain Python functions):
- `search_corpus(query: str, k: int = 5) -> list[Chunk]`
- `find_symbol(name: str) -> Chunk | None` — case-insensitive exact match
- `read_file(path: str, line_start: int, line_end: int) -> str` — max 100 lines, truncates with warning
- `get_class_outline(class_name: str) -> str` — own + inherited method signatures (BFS over base classes, depth 3, subclass override wins), unexpanded external bases, and direct subclasses. Answer node only. Replaced the hardcoded 'Base* -> Async*' prompt rule in task 6.5; cuts recorded outline calls 38 -> 26.

**Citations**: answer node emits `[short/path.py:start-end]` markers (corpus root stripped). Validated via `db.chunk_exists_at`; invalid markers stripped with footnote `"*N citation(s) could not be verified and were removed.*"`.

**Answer node tool loop**: `MAX_TOOL_ROUNDS = 8`. Claude calls `get_class_outline` (returns own + inherited methods via an MRO walk, plus external bases and direct subclasses) then `read_file` → result fed back → repeat until plain response or 8 rounds exhausted. Budget exhausted → forced final answer + `budget_exhausted=True` in `additional_kwargs`. Loop runs inside a single `answer_node` function (not separate graph nodes).

**Chunk context in system prompt**: full chunk `text` injected into system prompt alongside citation rule. User message = raw query only.

**Eval**: two sets. Dev = 50 questions in `evals/questions.jsonl` (prompt was tuned against the original 34). Held-out = 17 in `evals/questions-test.jsonl`, written from source after the prompt froze at `0061b3b`, never used for tuning. Both cover 7 tiers (recall / behavior / hard / definition / usage / cross-file / negative). Hybrid scoring: auto file-path check + LLM-as-judge (Claude Sonnet 4.6). N-run per question (default n=3); reports median score + variance. Results written to `evals/results/results-<mmdd-hhmm>-<agent>-<judge>.md`. Baseline: **91.0/95 (95.8%)** — dev set (50 q), n=3, `anthropic/claude-sonnet-5` agent+judge via OpenRouter, run 2026-08-12. Replaces the old 63/67 (94%, n=1, haiku, 34 questions, direct Anthropic) — different provider, model, and question count, so not a fair prior comparison. See `docs/plans/open-questions/milestone-6.md` A1 for the two bugs found investigating the non-perfect rows (one crash-handling fix, one eval-criteria fix) and why a clean re-run is still open. Held-out end-to-end: 32/32 (100%), n=1, same models, run 2026-08-12.

**Instrumentation (task 6.7):** every results file now reports per-tier p50/p95 latency, mean tool rounds, budget-exhaustion rate, and citation precision/strip-rate/hallucinated-path-rate (`evals/run.py: aggregate_by_tier`). `make eval` gates on `evals/baseline.json` and exits non-zero on a >2-point regression (`evals/run.py: main`). Judge validated against 40 blind-labeled samples: κ=0.872 clean (0.440 raw — 7/40 rows were rubric-drift from the A1 fix, not real disagreement). See `docs/plans/open-questions/milestone-6.md` A3 and `evals/judge_validation.py`.

**Held-out reversed two retrieval conclusions.** Dev has 31/33 questions naming their own gold symbol; held-out has 7/15. On held-out, RRF hybrid is the best config (93.3% vs dense 86.7%) and the symbol pre-check adds no recall over plain RRF (93.3% both), only MRR (0.880 vs 0.813). Do not quote the dev 97% as a retrieval number. See README and docs/plans/open-questions/milestone-6.md B1/B1b.

## Corpus

`langchain-core` only (`libs/core/` from `langchain-ai/langchain`). Pinned commit SHA in `indexer/corpus_config.py`.

Current pinned SHA: `1519ed5afbc3bfcc7170b12baa07f1ae7e98edd0` — 181 .py files, 2414 chunks.

## Implementation Notes

**sqlite-vec KNN syntax**: `WHERE embedding MATCH ? AND k = ?` — no LIMIT clause. Using LIMIT causes `OperationalError`. See `storage/db.py:vector_search`.

**Embed text truncation**: `indexer/embedder.py` truncates embed_text to `MAX_CHARS = 24_000` chars before sending to OpenAI to stay under 8192-token limit.

**Provider**: everything goes through OpenRouter via the OpenAI-compatible client (`ChatOpenAI` + `OpenAI`, both with `base_url=provider.BASE_URL`). There is no `anthropic` or `langchain-anthropic` dependency. Provider errors are `openai.APIError` — catching `anthropic.APIError` silently stops handling them.

**Embed model guard**: `indexer/embedder.py` writes `EMBED_MODEL` into the `meta` table; `retrieval/pipeline.py` calls `db.assert_embed_model()` on first use. An index predating the guard records nothing and is allowed through.

**Vector parity**: the name guard cannot catch same-name-different-vectors, which is the risk when swapping provider. `verify_embeddings.py` re-embeds already-stored chunks and cosines them against the index. Threshold 0.9999 — float32 storage alone costs ~1e-7, so a real match never lands near it. Exit 1 means re-index and re-run `make eval-retrieval`.

**Python env**: `.venv` has all deps (`sqlite_vec`, etc). Base anaconda3 env does not. Always use `.venv/bin/python -m pytest` for tests (macOS/Linux); `.venv/Scripts/python -m pytest` on Windows.

**BM25 score filter**: BM25Okapi IDF = 0 when a token appears in exactly half the corpus — do not filter by `score > 0`. Return top-k unconditionally. See `retrieval/bm25_index.py:search`.

## Writing Style

Prose (docs, PR bodies, commit bodies, error messages, comments) follows ASD-STE100 Simplified Technical English — rule set in global CLAUDE.md. For an audit or rewrite pass, use the `ste-writing` skill.

## Scope Boundary

**Not in scope**: symbol graph/DuckDB, `find_callers`/`find_callees`, reranker, 6-node graph, React/FastAPI, Docker, multi-provider LLM, Langfuse.
