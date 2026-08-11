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

Each query runs two searches:

- **BM25** — tokenizer expands camelCase and snake_case identifiers (`RunnableSequence` → `["runnable", "sequence", "runnablesequence"]`) so symbol names match even when the query uses different casing or word order. Costs ~1 ms per query over the in-memory index.
- **Dense** — OpenAI `text-embedding-3-small` cosine similarity over all 2414 chunk embeddings via `sqlite-vec`. The embedding request dominates query latency at ~300 ms.

The two run in sequence. Overlapping them on a thread would hide under 1% of the wall clock, so it is not worth the machinery.

Both return top-10 candidates. **Reciprocal Rank Fusion** (RRF, k=60) merges the lists by rank position rather than raw score — so the incompatible BM25 and cosine scales don't need normalisation. Top-5 chunks go to the agent.

### What the ablation says about that design

`make eval-retrieval` grades the retriever alone against hand-labeled gold symbols, with no LLM in the loop. 33 questions, binary relevance, a hit requires the right symbol in the right file.

| configuration | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 only | 39.4% | 54.5% | 0.293 | 0.300 |
| Dense only | 69.7% | 81.8% | 0.419 | 0.472 |
| RRF hybrid | 63.6% | 81.8% | 0.439 | 0.468 |
| **RRF + symbol pre-check** | **97.0%** | **97.0%** | **0.924** | **0.929** |

Two results worth stating plainly, because neither matches what this README claimed before it was measured:

1. **Hybrid fusion is not the thing that works.** RRF scores *below* dense-only at k=5 (63.6% against 69.7%) and ties it at k=10. BM25 ranks the correct chunk poorly enough that fusing it in costs more than it adds at the top of the list. RRF does improve MRR slightly (0.439 against 0.419), so it orders its hits better while finding fewer.
2. **The symbol pre-check does the real work.** It lifts recall@5 from 63.6% to 97.0%. It was described as a heuristic detail; it is the single highest-value component in the retriever.

**Read the 97% with its confound.** 31 of the 33 questions contain a gold symbol name verbatim, and the pre-check is exact symbol-name lookup, so the question set is close to purpose-built for it. Split by phrasing:

| configuration | names the symbol | does not name it |
|---|---|---|
| BM25 only | 13/31 | 0/2 |
| Dense only | 22/31 | 1/2 |
| RRF hybrid | 20/31 | 1/2 |
| RRF + symbol pre-check | 31/31 | 1/2 |

A set drawn mostly from "Where is X defined?" cannot separate a good retriever from a good string match, and two questions is not a sample. The held-out set in `docs/plans/milestone-6/task-6-4.md` adds questions phrased without the symbol name; until then, treat 97% as an upper bound for symbol-naming queries only.

### Agent

A 2-node LangGraph graph: `retrieve → answer`.

The retrieve node runs a heuristic pre-check: if the query contains a CamelCase or snake_case token that matches a known symbol name, `find_symbol` is called first to guarantee an exact-match chunk is included before `search_corpus` fills remaining slots.

The answer node runs a tool-call loop (max 8 rounds). Each round the LLM may call:
- `get_class_outline(class_name)` — returns all method signatures and line ranges for a class in one call, so the agent can plan which methods to read before issuing any `read_file` calls.
- `read_file(path, line_start, line_end)` — returns up to 100 lines of source.

### Citations

The agent is prompted to emit `[path:start-end]` markers for every factual claim. Before returning, each marker is checked against the index via `db.chunk_exists_at`. Markers that fail are stripped and a footnote is appended: `"*N citation(s) could not be verified and were removed.*"` Markers written against a longer path (`langchain_core/runnables/base.py`) are rewritten to the canonical short path rather than discarded.

**What the check proves:** the cited line range falls inside an indexed symbol in that file. **What it does not prove:** that those lines support the claim. It catches an invented file and an invented line range. It does not catch a real range cited for the wrong reason. Task 6.7 in `docs/plans/milestone-6/` adds the metric that measures the stronger property.

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
make check            # run unit tests
make eval             # 34-question end-to-end eval (LLM agent + LLM judge)
make eval-retrieval   # retriever-only ablation: recall@k, MRR, nDCG@5 (no LLM)
```

`make eval-retrieval` needs no Anthropic key and caches its query embeddings, so it runs in seconds and costs nothing after the first pass. Use it to check a retrieval change before spending a full eval.

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

Baseline: **63 / 67 (94%)** — haiku-4-5 agent, sonnet-4-6 judge, **n=1**, 34 questions.

Read that number with three caveats:

1. **n=1, so there is no variance figure.** The runner defaults to n=3 and reports a median and a variance. The published number is a single sample. Run `make eval` to regenerate at n=3.
2. **The system prompt was tuned against these 34 questions.** The score measures the prompt on the set that shaped it. It does not predict behavior on an unseen question. Task 6.4 adds a held-out set.
3. **Tier coverage is uneven.** Counts are behavior 11, hard 10, recall 9, and 1 each for definition, usage, cross-file, and negative. Per-tier numbers for those last four rest on a single question.

Results are written to `evals/results/results-<timestamp>-<agent>-<judge>.md`.

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph — 2 nodes: `retrieve → answer` |
| LLM | Claude (configurable via `AGENT_MODEL`, default `claude-haiku-4-5`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Storage | SQLite + `sqlite-vec` |
| BM25 | `rank-bm25` |
| UI | Streamlit |

## Limitations

- **Cold start.** The BM25 index is built in memory at the first query of each process. That costs ~380 ms over 2414 chunks and repeats on every restart. Nothing is persisted.
- **One session at a time.** `DB` holds a single SQLite connection in a module-level global with `check_same_thread=True`. A second concurrent Streamlit session raises `ProgrammingError`.
- **Symbol name collisions.** 1296 distinct symbol names cover 2414 chunks. `find_symbol` picks one chunk by a fixed tie-break rule (class, then method, then function, then path). It does not ask which `invoke` you meant.
- **Citation checking is containment-based.** See the Citations section above for the exact guarantee.

## What's not included

Symbol call graph, cross-file call tracing (`find_callers`/`find_callees`), reranker, React frontend, Docker, multi-provider LLM support.
