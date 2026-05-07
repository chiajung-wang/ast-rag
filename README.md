# ast-rag

Ask natural-language questions about `langchain-core` source code. Get answers with citations that link directly to the relevant lines on GitHub.

```
> Where is RunnableSequence defined?

RunnableSequence is defined in [runnables/base.py:891-1205]. It is the core
composition primitive — created when you chain runnables with the `|` operator.
[runnables/base.py:891-1205]
```

## How it works

1. **Index** — `langchain-core` source is parsed with Python's `ast` module into chunks (one per function, class, and method). Chunks are embedded with OpenAI `text-embedding-3-small` and stored in SQLite with `sqlite-vec`.
2. **Retrieve** — queries run hybrid BM25 + dense vector search, merged via reciprocal rank fusion into top-5 results.
3. **Answer** — a 2-node LangGraph agent (Claude Sonnet 4.6) generates answers with `[file:line_start-line_end]` citation markers, validated against the index before returning.
4. **UI** — Streamlit chat interface with expandable citation blocks showing source lines and GitHub permalinks.

## Setup

```bash
# Install dependencies
make install

# Index langchain-core (one-time, ~$0.50 in OpenAI API calls)
make index

# Launch the UI
make run
```

Requires `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your environment. Set `AGENT_MODEL` to override the default LLM (default: `claude-haiku-4-5`).

## CLI usage

```bash
# Ask a question directly
python ask.py "Where is Runnable defined?"

# Test raw retrieval
python query.py "Runnable definition"

# Run evaluation
make eval
```

## Status

| Milestone | What | Status |
|---|---|---|
| 1 | Foundation & Indexer — storage, AST chunker, embedder, clone script, pipeline | ✅ Complete |
| 2 | Hybrid Retrieval — BM25 + dense RRF | ✅ Complete |
| 3 | LangGraph Agent — retrieve + answer nodes | ✅ Complete |
| 4 | Streamlit UI + citations | Planned |
| 5 | Eval harness | Planned |

Index: 2414 chunks from `langchain-core` at commit `1519ed5a`.

## Eval results

_Run `make eval` to generate._

<!-- evals/results.md is generated and not committed. Run make eval locally. -->

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph — 2 nodes: `retrieve → answer` |
| LLM | Claude (configurable via `AGENT_MODEL`, default `claude-haiku-4-5`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Storage | SQLite + `sqlite-vec` |
| BM25 | `rank-bm25` |
| UI | Streamlit |

## What's not included

Symbol call graph, cross-file call tracing (`find_callers`/`find_callees`), reranker, React frontend, Docker, multi-provider LLM support.
