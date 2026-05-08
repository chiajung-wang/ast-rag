# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ast-rag` — RAG-over-code for `langchain-core`. Ask natural-language questions about the codebase; get answers with validated, source-linked citations.

## Commands

```bash
make install   # install deps (uv or pip)
make run       # launch Streamlit UI
make index     # (re)build SQLite index from langchain-core source
make eval      # run 34-question eval, write timestamped results to evals/results/

python ask.py "Where is Runnable defined?"   # CLI agent test
python query.py "Runnable definition"        # raw retrieval test
```

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph (Python) — 2 nodes: `retrieve → answer` |
| LLM | Claude — model set via `AGENT_MODEL` env var (default: `claude-haiku-4-5`) |
| Embeddings | OpenAI `text-embedding-3-small` |
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

**Hybrid Retrieval**: BM25 top-10 + dense top-10 → reciprocal rank fusion → top-5. BM25 tokenizer expands camelCase + snake_case (`RunnableSequence` → `["runnable", "sequence", "runnablesequence"]`, `invoke_async` → `["invoke", "async", "invoke_async"]`). Implemented in `retrieval/bm25_index.py`, `retrieval/rrf.py`, `retrieval/pipeline.py`.

**Retrieve node logic**: heuristic pre-check extracts CamelCase/snake_case tokens from query, checks symbol name set. Match → `find_symbol` first, then `search_corpus` for remaining slots. No match → `search_corpus` only.

**Tools** (plain Python functions):
- `search_corpus(query: str, k: int = 5) -> list[Chunk]`
- `find_symbol(name: str) -> Chunk | None` — case-insensitive exact match
- `read_file(path: str, line_start: int, line_end: int) -> str` — max 100 lines, truncates with warning
- `get_class_outline(class_name: str) -> str` — returns all method signatures + line ranges for a class (answer node only)

**Citations**: answer node emits `[short/path.py:start-end]` markers (corpus root stripped). Validated via `db.chunk_exists_at`; invalid markers stripped with footnote `"*N citation(s) could not be verified and were removed.*"`.

**Answer node tool loop**: `MAX_TOOL_ROUNDS = 8`. Claude calls `get_class_outline` (batches async sibling + all mixins before any reads) then `read_file` → result fed back → repeat until plain response or 8 rounds exhausted. Budget exhausted → forced final answer + `budget_exhausted=True` in `additional_kwargs`. Loop runs inside a single `answer_node` function (not separate graph nodes).

**Chunk context in system prompt**: full chunk `text` injected into system prompt alongside citation rule. User message = raw query only.

**Eval**: 34 hand-crafted questions in `evals/questions.jsonl` across 7 tiers (recall / behavior / hard / definition / usage / cross-file / negative). Hybrid scoring: auto file-path check + LLM-as-judge (Claude Sonnet 4.6). N-run per question (default n=3); reports median score + variance. Results written to `evals/results/results-<mmdd-hhmm>-<agent>-<judge>.md`. Baseline: 63/67 (94%) at n=1.

## Corpus

`langchain-core` only (`libs/core/` from `langchain-ai/langchain`). Pinned commit SHA in `indexer/corpus_config.py`.

Current pinned SHA: `1519ed5afbc3bfcc7170b12baa07f1ae7e98edd0` — 181 .py files, 2414 chunks.

## Implementation Notes

**sqlite-vec KNN syntax**: `WHERE embedding MATCH ? AND k = ?` — no LIMIT clause. Using LIMIT causes `OperationalError`. See `storage/db.py:vector_search`.

**Embed text truncation**: `indexer/embedder.py` truncates embed_text to `MAX_CHARS = 24_000` chars before sending to OpenAI to stay under 8192-token limit.

**Python env**: `.venv` has all deps (`sqlite_vec`, etc). Base anaconda3 env does not. Always use `.venv/Scripts/python -m pytest` for tests.

**BM25 score filter**: BM25Okapi IDF = 0 when a token appears in exactly half the corpus — do not filter by `score > 0`. Return top-k unconditionally. See `retrieval/bm25_index.py:search`.

## Scope Boundary

**Not in scope**: symbol graph/DuckDB, `find_callers`/`find_callees`, reranker, 6-node graph, React/FastAPI, Docker, multi-provider LLM, Langfuse.
