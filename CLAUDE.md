# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ast-rag` — RAG-over-code for `langchain-core`. Ask natural-language questions about the codebase; get answers with validated, source-linked citations.

## Commands

```bash
make install   # install deps (uv or pip)
make run       # launch Streamlit UI
make index     # (re)build SQLite index from langchain-core source
make eval      # run 20-question eval, write evals/results.md

python ask.py "Where is Runnable defined?"   # CLI agent test
python query.py "Runnable definition"        # raw retrieval test
```

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph (Python) — 2 nodes: `retrieve → answer` |
| LLM | Claude Sonnet 4.6 (`claude-sonnet-4-6`) |
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
  retrieve → answer (Claude Sonnet 4.6)
       ↓
  Tools: search_corpus / find_symbol / read_file
       ↓
  Streamlit chat UI
```

Entire system runs in one Python process. No services, no Docker, no separate frontend.

## Key Implementation Details

**Chunk granularity**: one chunk per top-level function, top-level class, and method. Methods are sibling chunks (not nested). Method embed text prefixed with `"ClassName.method_name: "` for BM25 and dense retrieval.

**Hybrid Retrieval**: BM25 top-10 + dense top-10 → reciprocal rank fusion → top-5. BM25 tokenizer expands camelCase (`RunnableSequence` → `["Runnable", "Sequence", "RunnableSequence"]`).

**Retrieve node logic**: heuristic pre-check extracts CamelCase/snake_case tokens from query, checks symbol name set. Match → `find_symbol` first, then `search_corpus` for remaining slots. No match → `search_corpus` only.

**Tools** (plain Python functions):
- `search_corpus(query: str, k: int = 5) -> list[Chunk]`
- `find_symbol(name: str) -> Chunk | None` — case-insensitive exact match
- `read_file(path: str, line_start: int, line_end: int) -> str` — max 100 lines, truncates with warning

**Citations**: answer node emits `[short/path.py:start-end]` markers (corpus root stripped). Validated before return; invalid markers stripped with footnote.

**Eval**: 20 hand-crafted questions in `evals/questions.jsonl`. Hybrid scoring: auto file-path check + LLM-as-judge. Results in `evals/results.md`.

## Corpus

`langchain-core` only (`libs/core/` from `langchain-ai/langchain`). Pinned commit SHA in `indexer/corpus_config.py`.

## Scope Boundary

**Not in scope**: symbol graph/DuckDB, `find_callers`/`find_callees`, reranker, 6-node graph, React/FastAPI, Docker, multi-provider LLM, Langfuse.
