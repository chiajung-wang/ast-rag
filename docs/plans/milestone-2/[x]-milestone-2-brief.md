# Milestone 2: Retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid BM25 + dense retrieval with RRF fusion and expose `search_corpus`, `find_symbol`, `read_file` as plain Python functions, verified by `python query.py`.

**Architecture:** BM25 index loaded in-memory from all chunk `embed_text` values. Dense search via `sqlite-vec` KNN. RRF merges BM25 top-10 + dense top-10 → top-5. `find_symbol` does case-insensitive exact match on `symbol_name`. `read_file` reads from corpus files with a 100-line cap.

**Tech Stack:** `rank-bm25`, `sqlite-vec`, Python `re` (camelCase tokenizer + symbol heuristic)

**Prerequisite:** Milestone 1 complete. `index.db` exists.

---

## Completion Criteria

1. All unit tests pass: `pytest tests/ -v`.
2. `python query.py "RunnableSequence"` — top result has `symbol_name == "RunnableSequence"`.
3. `python query.py "how does streaming work"` — returns ≥3 chunks with `symbol_type` of `"method"` or `"function"`.
4. `find_symbol("RunnablePassthrough")` returns the class chunk in <50 ms.
5. `read_file("runnables/base.py", 1, 10)` returns 10 lines of source.
6. `read_file("runnables/base.py", 1, 500)` returns exactly 100 lines + truncation notice.

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 2.1 | BM25 index | `retrieval/bm25_index.py`, `tests/test_bm25.py` |
| 2.2 | RRF fusion | `retrieval/rrf.py`, `tests/test_rrf.py` |
| 2.3 | Retrieval tools | `retrieval/pipeline.py`, `tests/test_pipeline.py` |
| 2.4 | `query.py` CLI + spot checks | `query.py` (update), smoke-test script |

---

## What Milestone 3 Depends On

- `search_corpus(query: str, k: int = 5) -> list[Chunk]`
- `find_symbol(name: str) -> Chunk | None`
- `read_file(path: str, line_start: int, line_end: int) -> str`
- `retrieve(query: str, k: int = 5) -> list[Chunk]` (heuristic symbol pre-check)
