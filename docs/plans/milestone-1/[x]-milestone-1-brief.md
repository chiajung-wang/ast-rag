# Milestone 1: Foundation + Indexer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the indexer pipeline so that `make index` populates a `.db` file from `langchain-core` source and `python query.py "Runnable"` returns relevant chunks.

**Architecture:** Clone `langchain-core` at a pinned commit. Walk every `.py` file with Python's `ast` module, emitting one Chunk per top-level function, class, and method. Embed chunks with OpenAI `text-embedding-3-small`. Store chunk metadata + embeddings in a single SQLite file via `sqlite-vec`.

**Tech Stack:** Python 3.11, `openai`, `sqlite-vec`, Python stdlib `ast`, `hashlib`

---

## Completion Criteria

1. `make install` completes without error on a clean Python 3.11 environment.
2. `make index` runs end-to-end and creates `index.db`.
3. `python query.py "RunnableSequence"` returns ≥1 chunk where `symbol_name == "RunnableSequence"`.
4. `python query.py "invoke method"` returns ≥1 chunk where `symbol_type == "method"`.
5. Index contains >1 000 chunks (langchain-core has ~50k LOC across ~300 files).
6. Re-running `make index` skips already-embedded chunks (idempotent — no duplicate API calls).
7. All unit tests pass: `pytest tests/ -v`.

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 1.1 | Project scaffold | `pyproject.toml`, `Makefile`, folder skeleton, `conftest.py` |
| 1.2 | Chunk dataclass + corpus config | `storage/chunk.py`, `indexer/corpus_config.py` |
| 1.3 | AST chunker | `indexer/chunker.py`, `tests/test_chunker.py` |
| 1.4 | Storage layer | `storage/db.py`, `tests/test_db.py` |
| 1.5 | Embedder | `indexer/embedder.py`, `tests/test_embedder.py` |
| 1.6 | Clone script | `indexer/clone.py` |
| 1.7 | Indexer pipeline + `query.py` | `indexer/__main__.py`, `query.py` |

---

## What Milestone 2 Depends On

- `Chunk` dataclass with all fields
- `DB` with `insert_chunk`, `insert_embedding`, `vector_search`, `symbol_lookup`, `all_chunks`, `all_symbol_names`, `chunk_exists_at`
- `chunk_corpus(corpus_root: Path) -> list[Chunk]`
- Populated `index.db` on disk
