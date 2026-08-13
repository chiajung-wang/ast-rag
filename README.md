# ast-rag

[![check](https://github.com/chiajung-wang/ast-rag/actions/workflows/check.yml/badge.svg)](https://github.com/chiajung-wang/ast-rag/actions/workflows/check.yml)

Ask natural-language questions about `langchain-core` source code. Get answers with citations that link directly to the relevant lines on GitHub.

```
> Where is RunnableSequence defined?

RunnableSequence is defined in [runnables/base.py:891-1205]. It is the core
composition primitive — created when you chain runnables with the `|` operator.
[runnables/base.py:891-1205]
```

## How it works

1. **Index** — `langchain-core` source is parsed with Python's `ast` module into chunks (one per function, class, and method). Chunks are embedded with `openai/text-embedding-3-small` via OpenRouter and stored in SQLite with `sqlite-vec`.
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
- **Dense** — `openai/text-embedding-3-small` cosine similarity over all 2414 chunk embeddings via `sqlite-vec`. The embedding request dominates query latency at ~300 ms.

The two run in sequence. Overlapping them on a thread would hide under 1% of the wall clock, so it is not worth the machinery.

Both return top-10 candidates. **Reciprocal Rank Fusion** (RRF, k=60) merges the lists by rank position rather than raw score — so the incompatible BM25 and cosine scales don't need normalisation. Top-5 chunks go to the agent.

### What the ablation says about that design

`make eval-retrieval` grades the retriever alone against hand-labeled gold symbols, with no LLM in the loop. 45 questions, binary relevance, a hit requires the right symbol in the right file.

| configuration | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 only | 42.2% | 60.0% | 0.357 | 0.352 |
| Dense only | 73.3% | 86.7% | 0.494 | 0.536 |
| RRF hybrid | 68.9% | 84.4% | 0.502 | 0.531 |
| **RRF + symbol pre-check** | **95.6%** | **97.8%** | **0.908** | **0.912** |

45 questions, up from the 33 originally measured in task 6.3 — task 6.4 added 12 dev questions without a re-run (filed as A5), refreshed 2026-08-12. Direction is unchanged; recall@5 moved a few points in both directions.

**38 of those 45 questions name their gold symbol verbatim**, and the pre-check is exact symbol-name lookup. So the set is close to purpose-built for it. The held-out set exists to check that, and it changes the answer:

| configuration | dev recall@5 | held-out recall@5 |
|---|---|---|
| BM25 only | 42.2% | 73.3% |
| Dense only | 73.3% | 86.7% |
| RRF hybrid | 68.9% | **93.3%** |
| RRF + symbol pre-check | **95.6%** | 93.3% |

Held-out is 15 graded questions, 7 of which name their symbol — a near-even split against the dev set's 31/33. Three conclusions, in order of how much weight they carry:

1. **The dev set's verdict on RRF was an artifact of phrasing.** On dev, RRF looked *worse* than dense alone (68.9% against 73.3%). On held-out it is the best configuration (93.3% against 86.7%). BM25 nearly doubles once questions stop being "Where is X defined?", which is where a lexical index should have been strong all along.
2. **The pre-check adds no recall on balanced phrasing.** On held-out it scores identically to plain RRF (93.3%), and identically inside both halves of the split: 7/7 naming, 7/8 not naming. What it does add is ranking — MRR 0.880 against 0.813. Its 97% on dev is a symbol-lookup score, not a retrieval score.
3. **Neither set is large enough to settle the architecture.** At n=15, the 93.3% against 86.7% gap is a single question. Treat the direction as a signal and the magnitude as noise.

The remaining held-out miss for both top configurations is `t11`, which asks what stops the same prompt reaching a model twice — the answer spans `caches.py` and `globals.py` and names neither symbol.

### Agent

A 2-node LangGraph graph: `retrieve → answer`.

The retrieve node runs a heuristic pre-check: if the query contains a CamelCase or snake_case token that matches a known symbol name, `find_symbol` is called first to guarantee an exact-match chunk is included before `search_corpus` fills remaining slots.

The answer node runs a tool-call loop (max 8 rounds). Each round the LLM may call:
- `get_class_outline(class_name)` — maps a class in one call: methods it defines, methods it inherits (breadth-first over base classes, each tagged with its defining class), base classes outside the corpus, and direct subclasses.

  Inheritance is not a nicety here. `BaseCallbackHandler` defines only 7 `ignore_*` flags — every `on_*` event lives on one of 6 mixins it inherits, and the async variants live on a subclass. The outline used to return 8 rows and no events, and the system prompt made up the difference with a hardcoded rule naming langchain-core classes. It now returns 29 rows including 20 events, the prompt rule is gone, and replaying the recorded eval traces needs 26 outline calls instead of 38.
- `read_file(path, line_start, line_end)` — returns up to 100 lines of source.

### Citations

The agent is prompted to emit `[path:start-end]` markers for every factual claim. Before returning, each marker is checked against the index via `db.chunk_exists_at`. Markers that fail are stripped and a footnote is appended: `"*N citation(s) could not be verified and were removed.*"` Markers written against a longer path (`langchain_core/runnables/base.py`) are rewritten to the canonical short path rather than discarded.

**What the check proves:** the cited line range falls inside an indexed symbol in that file. **What it does not prove:** that those lines support the claim. It catches an invented file and an invented line range. It does not catch a real range cited for the wrong reason. Task 6.7 in `docs/plans/milestone-6/` adds the metric that measures the stronger property.

### Eval

Two hand-written sets across 7 tiers (recall, behavior, hard, definition, usage, cross-file, negative): 50 dev questions and 17 held-out. Each answer is scored 0–2:

- **+1** if the expected file path appears in the answer (objective, free)
- **+1** if an LLM judge (Claude Sonnet 4.6) rates the answer as passing

Negative-tier questions cap at 1 (no file path check — model must correctly say the answer is not in corpus).

## Setup

```bash
# Install dependencies
make install

# Index langchain-core (one-time, ~$0.50 in embedding calls)
make index

# Launch the UI
make run
```

Requires a single `OPENROUTER_API_KEY`. Copy `.env.example` to `.env` and fill it in — OpenRouter serves both the chat models and the embeddings, so there is one key and one bill.

| var | default | notes |
|---|---|---|
| `OPENROUTER_API_KEY` | — | required, https://openrouter.ai/keys |
| `EMBED_MODEL` | `openai/text-embedding-3-small` | **changing this invalidates the index** — see below |
| `AGENT_MODEL` | `anthropic/claude-haiku-4.5` | answering agent |
| `JUDGE_MODEL` | `anthropic/claude-sonnet-4.6` | eval judge only |

Run `python verify_embeddings.py` after changing the provider or the embedding model. It re-embeds chunks that already have stored vectors and compares them against the index, so it catches the case the name check cannot: same model name, different vectors. Exit 0 means the index is safe to keep.

**On `EMBED_MODEL`:** embeddings from two different models do not share a vector space, so a query embedded by one cannot be compared against an index built by the other. Nothing about that fails on its own — retrieval just returns plausible, wrong neighbours. The index records the model it was built with, and a query under a different `EMBED_MODEL` raises `EmbedModelMismatch` rather than degrading quietly. Re-run `make index` after changing it.

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
make eval             # 50-question dev-set end-to-end eval (LLM agent + LLM judge)
make eval-test        # 17-question held-out end-to-end eval (never used for tuning)
make eval-retrieval   # retriever-only ablation: recall@k, MRR, nDCG@5 (no LLM)
```

`make eval-retrieval` makes no chat-model calls and caches its query embeddings, so it runs in seconds and costs nothing after the first pass. Use it to check a retrieval change before spending a full eval.

CI (`.github/workflows/check.yml`) runs `make check` on every push and pull request. It does not run the two `make eval` targets — those spend API budget and are a manual, explicit step.

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
| 6 | Maintenance — retrieval/citation bug fixes, held-out eval set, class-outline inheritance walk, move to OpenRouter, CI, prompt caching, eval instrumentation | ✅ Complete |

Index: 2414 chunks from `langchain-core` at commit `1519ed5a`.

## Eval results

Two question sets:

| set | file | questions | purpose |
|---|---|---|---|
| dev | `evals/questions.jsonl` | 50 | Iteration. The system prompt was tuned against the original 34. |
| held-out | `evals/questions-test.jsonl` | 17 | Written from source *after* the prompt was frozen at `0061b3b`. Never used for tuning. |

Tier counts are now 5 or more per tier on both sets. The original dev set had 1 question each in definition, usage, cross-file, and negative.

**End-to-end dev baseline: 91.0 / 95 (95.8%)** — dev set (50 q), n=3, `anthropic/claude-sonnet-5` agent+judge via OpenRouter, run 2026-08-12, $9.96. Replaces an earlier 63/67 (94%) figure that was n=1, haiku-4-5, direct Anthropic, and only 34 questions — not directly comparable on any axis.

**End-to-end held-out: 32 / 32 (100%)** — held-out set (17 q), n=1, same models, run 2026-08-12, $1.00. This is the score that matters more: written after the prompt froze, never used for tuning.

Investigating the dev run's non-perfect rows found two bugs, both fixed: a silent crash where a non-`openai.APIError` SDK exception escaped the agent's error handling uncaught (now caught generically), and 3 grading-criteria bugs in the original 34-question set that demanded facts absent from the indexed corpus, contradicting the system prompt's own citation rule (now corrected). A clean re-run reflecting both fixes has not been done — see `docs/plans/open-questions/milestone-6.md` A1 for the honest accounting.

**Judge agreement: κ = 0.872** (Cohen's kappa, 33 blind human-labeled samples, clears the ≥0.8 bar). Raw agreement over the full 40-sample draw was 0.440 — misleadingly low, because 7 of the 40 rows were pulled from *before* the q28/q29/q30 grading-criteria fix above and the human correctly labeled against the already-corrected rubric. Excluding those, the human and the original judge agree on 32 of 33 (97.0%). The one real disagreement — whether an answer that lists a class's fields and methods but never states its purpose as an explicit sentence should pass — was independently flagged the same way by a completely different model (`anthropic/claude-haiku-4.5`) re-judging the same sample (39/40, 97.5%, same rubric-drift caveat). Full accounting, including the exact disagreement, in `docs/plans/open-questions/milestone-6.md` A3.

`make eval` now gates on `evals/baseline.json` and exits non-zero on a regression of more than 2 points versus the last committed baseline (currently 91.0/95 dev, 32/32 held-out). Every results file also reports p50/p95 latency, mean tool rounds, budget-exhaustion rate, and three citation-quality figures (precision, strip rate, hallucinated-path rate) per tier.

Results go to `evals/results/results-<timestamp>-<set>-<agent>-<judge>.md`, with the question set named in the filename so a dev score and a held-out score cannot be confused.

## Stack

| Layer | Choice |
|---|---|
| Agent | LangGraph — 2 nodes: `retrieve → answer` |
| Provider | OpenRouter — one key for chat and embeddings |
| LLM | `AGENT_MODEL`, default `anthropic/claude-haiku-4.5` |
| Embeddings | `EMBED_MODEL`, default `openai/text-embedding-3-small` |
| Storage | SQLite + `sqlite-vec` |
| BM25 | `rank-bm25` |
| UI | Streamlit |

## Limitations

- **Cold start.** The BM25 index is built in memory at the first query of each process. That costs ~380 ms over 2414 chunks and repeats on every restart. Nothing is persisted.
- **Prompt caching is size-dependent.** The system prompt is marked as one cacheable block, but Haiku 4.5 will not cache a prefix under 4096 tokens and gives no signal when it declines. Measured across 5 sample questions this prompt runs 3,395 to 29,231 tokens (median 7,319), so most questions cache and small-chunk ones do not. `cache_read_tokens` is reported per run so the effect is measured rather than assumed.
- **Concurrency is read-only.** `DB` holds one SQLite connection in a module-level global, opened with `check_same_thread=False` so a second Streamlit session does not raise. Every query path is a read; the indexer writes single-threaded.
- **Symbol name collisions.** 1299 distinct symbol names cover 2414 chunks. `find_symbol` picks one chunk by a fixed tie-break rule (class, then method, then function, then path). It does not ask which `invoke` you meant.
- **Citation checking is containment-based.** See the Citations section above for the exact guarantee.

## What's not included

Symbol call graph, cross-file call tracing (`find_callers`/`find_callees`), reranker, React frontend, Docker, multi-provider LLM support.
