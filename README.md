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
3. **Answer** — a 2-node LangGraph agent (Claude Haiku 4.5 by default) generates answers with `[file:line_start-line_end]` citation markers, validated against the index before returning.
4. **UI** — Streamlit chat interface with expandable citation blocks showing source lines and GitHub permalinks.

## Technical details

### Chunking

Source is parsed with Python's `ast` module — one chunk per top-level function, top-level class, and method. Methods are stored as **sibling chunks** (not nested inside the class chunk) to avoid oversized blobs that dilute embedding signal.

Method embed text is prefixed with `"ClassName.method_name: "` before indexing so dense retrieval can distinguish `Runnable.invoke` from `BaseTool.invoke` without extra context at query time.

### Hybrid retrieval

Each query runs two searches in parallel:

- **BM25** — tokenizer expands camelCase and snake_case identifiers (`RunnableSequence` → `["runnable", "sequence", "runnablesequence"]`) so symbol names match even when the query uses different casing or word order.
- **Dense** — OpenAI `text-embedding-3-small` cosine similarity over all 2414 chunk embeddings via `sqlite-vec`.

Both return top-10 candidates. **Reciprocal Rank Fusion** (RRF, k=60) merges the lists by rank position rather than raw score — so the incompatible BM25 and cosine scales don't need normalisation. Top-5 chunks go to the agent.

### Agent

A 2-node LangGraph graph: `retrieve → answer`.

The retrieve node runs a heuristic pre-check: if the query contains a CamelCase or snake_case token that matches a known symbol name, `find_symbol` is called first to guarantee an exact-match chunk is included before `search_corpus` fills remaining slots.

The answer node runs a tool-call loop (max 8 rounds). Each round the LLM may call:
- `get_class_outline(class_name)` — returns all method signatures and line ranges for a class in one call, so the agent can plan which methods to read before issuing any `read_file` calls.
- `read_file(path, line_start, line_end)` — returns up to 100 lines of source.

### Citations

The agent is prompted to emit `[path:start-end]` markers for every factual claim. Before returning, each marker is validated against the index via `db.chunk_exists_at`. Invalid markers are stripped and a footnote is appended: `"*N citation(s) could not be verified and were removed.*"`

### Eval

34 hand-crafted questions across 7 tiers (recall, behavior, hard, definition, usage, cross-file, negative). Each answer is scored 0–2:

- **+1** if the expected file path appears in the answer (objective, free)
- **+1** if an LLM judge (Claude Sonnet 4.6) rates the answer as passing

Negative-tier questions cap at 1 (no file path check — model must correctly say the answer is not in corpus).

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

### macOS note

`make index` requires a Python with `enable_load_extension` support for `sqlite-vec`. The Python.org macOS installer disables this by default. If you see:

```
AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'
```

Recreate the venv with Homebrew Python (which has it enabled):

```bash
brew install python@3.11
rm -rf .venv
uv venv --python $(brew --prefix python@3.11)/bin/python3.11
uv sync
```

Or use uv's managed Python:

```bash
rm -rf .venv
uv python install 3.11
uv venv --python 3.11
uv sync
```

## Development

```bash
make check   # run unit tests
make eval    # run 34-question eval, write results to evals/results/
```

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
| 4 | Streamlit UI + citations + eval harness | ✅ Complete |
| 5 | Polish + ship — error handling, make check, eval results | ✅ Complete |

Index: 2414 chunks from `langchain-core` at commit `1519ed5a`.

## Eval results

Baseline: **63 / 67 (94%)** — haiku-4-5 agent, sonnet-4-6 judge, n=1 across 34 questions (7 tiers: recall / behavior / hard / definition / usage / cross-file / negative).

Run `make eval` to regenerate. Results written to `evals/results/results-<timestamp>-<agent>-<judge>.md`.

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
