# PRD: ast-rag MVP

**Status:** Draft | **Timeline:** 7 days | **Scope:** `langchain-core` only

---

## Problem Statement

A developer building a production RAG-over-code system (DayOne v1) needs to validate core technical bets — AST chunking quality, hybrid retrieval accuracy, LangGraph tool-calling mechanics, and eval scoring methodology — before committing 6 weeks of work. Without a working prototype, these risks are unknown and could cause DayOne v1 to fail or slip significantly.

Additionally, the developer needs a credible, shippable "I can build this" artifact for job applications within 7 days, before DayOne v1 is complete.

---

## Solution

`ast-rag` is a 7-day, single-Python-process prototype that:

1. Indexes `langchain-core` (~50k LOC) using AST-based chunking into SQLite with vector search.
2. Retrieves relevant code chunks via hybrid BM25 + dense embedding search with reciprocal rank fusion.
3. Answers natural-language questions about `langchain-core` through a 2-node LangGraph agent (Claude Sonnet 4.6) that emits validated citations.
4. Exposes a Streamlit chat UI with expandable citation links.
5. Evaluates itself against 20 hand-crafted questions with published scores.

---

## User Stories

### Indexing

1. As a developer, I want to clone `langchain-core` at a pinned commit, so that the corpus is reproducible across machines.
2. As a developer, I want the indexer to parse every `.py` file with Python's `ast` module, so that I get function/class granularity chunks.
3. As a developer, I want each chunk to carry `{file_path, symbol_name, symbol_type, line_start, line_end, docstring}` metadata, so that the agent can produce precise, verifiable citations.
4. As a developer, I want chunks embedded with `text-embedding-3-small`, so that semantic queries work without large infrastructure costs (~$0.50 total).
5. As a developer, I want embeddings stored in SQLite via `sqlite-vec`, so that the entire index is a single portable `.db` file requiring zero external services.
6. As a developer, I want to rebuild the index with a single `make index` command, so that re-indexing after changes is trivial.
7. As a developer, I want to spot-check the index via `python query.py "<query>"`, so that I can verify indexing worked before running the full agent.

### Retrieval

8. As a developer, I want BM25 retrieval over all chunks using `rank-bm25`, so that keyword-exact queries (e.g., class names) always surface the right chunk.
9. As a developer, I want dense vector retrieval returning the top-10 nearest neighbors, so that semantic queries work for concept-level questions.
10. As a developer, I want BM25 top-10 and dense top-10 merged via reciprocal rank fusion to produce top-5 results, so that hybrid retrieval outperforms either alone without a reranker.
11. As a developer, I want `search_corpus(query, k=5)` as a plain Python function, so that the agent and tests can call retrieval directly without LangGraph boilerplate.
12. As a developer, I want `find_symbol(name)` to do exact-match lookup by symbol name, so that the agent can resolve a known class or function name to its definition in one call.
13. As a developer, I want `read_file(path, line_start, line_end)` to return raw source lines, so that the agent can fetch extended context around a citation when needed.

### Agent

14. As a developer, I want a 2-node LangGraph graph (`retrieve → answer`), so that I learn the LangGraph API on the smallest possible graph before scaling to 6 nodes in DayOne v1.
15. As a developer, I want the `retrieve` node to call `search_corpus` and optionally `find_symbol` based on the query, so that the agent always has relevant context before generating an answer.
16. As a developer, I want the `answer` node to be powered by Claude Sonnet 4.6, so that the model is consistent with DayOne v1's hardcoded choice.
17. As a developer, I want the `answer` node to produce responses with `[file:line_start-line_end]` citation markers, so that every factual claim is traceable to source code.
18. As a developer, I want the `answer` node to call `read_file` when it needs additional lines beyond the retrieved chunk, so that answers to multi-line context questions remain accurate.
19. As a developer, I want a one-paragraph system prompt specifying persona, citation requirement, and uncertainty instruction, so that the model's behavior is predictable without prompt engineering overhead.
20. As a developer, I want the agent callable from a CLI script via `python ask.py "<question>"`, so that I can iterate on agent behavior without launching Streamlit.

### Citation Validation

21. As a developer, I want citation markers parsed out of every answer before it is returned, so that I can verify each cited location exists in the index.
22. As a developer, I want invalid citations (file or line range not in index) stripped from the answer with a footnote noting the removal, so that users are never shown a citation that doesn't resolve.
23. As a developer, I want valid citations to survive the validation step unchanged, so that accurate answers are not degraded.

### UI

24. As a developer using the Streamlit UI, I want a text input box for questions, so that I can query the agent without a CLI.
25. As a developer, I want message history displayed in the Streamlit session, so that I can review earlier answers in the same session.
26. As a developer, I want agent answers rendered as Markdown, so that code blocks, bullet lists, and emphasis display correctly.
27. As a developer, I want each citation to render as a Markdown link that expands to show the cited source lines, so that I can verify citations without leaving the UI.

### Evaluation

28. As a developer, I want 20 hand-crafted evaluation questions in `evals/questions.jsonl` with `{question, expected_file_path, ground_truth_answer}` fields, so that the eval set covers easy, medium, and hard query types.
29. As a developer, I want the eval set split into 8 easy (single-symbol lookup), 8 medium (concept explanation), and 4 hard (multi-step / contextual) questions, so that scores are diagnostic rather than just a single number.
30. As a developer, I want each question scored 0/1/2 against ground truth, so that partial credit is possible and the scoring differentiates near-misses from failures.
31. As a developer, I want eval results written to `evals/results.md` with per-question scores, a baseline total, and a post-iteration total, so that improvement from tuning is documented.
32. As a developer, I want to run the full eval with `make eval`, so that re-running after changes is a single command.

### Developer Experience

33. As a developer, I want `make install` to install all dependencies via `uv` or `pip`, so that onboarding from a clean environment is one command.
34. As a developer, I want `make run` to launch the Streamlit UI, so that running the app is discoverable from the Makefile.
35. As a developer, I want a `README.md` that documents what the project does, how to run it, the eval result, and what is deliberately out of scope, so that the repo is self-explanatory to a recruiter or interviewer.

---

## Implementation Decisions

### Modules

**Indexer** (`indexer/`): One-time pipeline. Clones corpus, runs AST chunker, embeds via OpenAI, writes to SQLite. Exposes a `build_index()` entry point called by `make index`.

**AST Chunker** (`indexer/chunker.py`): Walks each `.py` file with `ast.parse`, emits one `Chunk` dataclass per top-level function or class. Metadata: `file_path`, `symbol_name`, `symbol_type` (`"function"` | `"class"`), `line_start`, `line_end`, `docstring` (first string literal in body, or `None`).

**Embedder** (`indexer/embedder.py`): Batches chunks, calls `openai.embeddings.create` with `text-embedding-3-small`, writes vectors to `sqlite-vec` table. Idempotent — skips already-embedded chunks by hash.

**Storage** (`storage/db.py`): Thin wrapper over `sqlite3` + `sqlite-vec`. Exposes: `insert_chunk`, `get_chunk_by_id`, `vector_search(embedding, k)`, `symbol_lookup(name)`, `line_range_read(path, start, end)`.

**BM25 Index** (`retrieval/bm25_index.py`): Loads all chunk texts at startup into `rank_bm25.BM25Okapi`. Returns scored candidate list. Rebuilt in-memory each run (corpus is small enough).

**Retrieval Pipeline** (`retrieval/pipeline.py`): Implements `search_corpus`, `find_symbol`, `read_file`. Combines BM25 + dense via RRF. This is the module the agent tools wrap.

**Agent** (`agent/graph.py`): LangGraph `StateGraph` with two nodes. State carries `messages` (LangChain message list) and `retrieved_chunks`. `retrieve` node populates `retrieved_chunks`; `answer` node invokes Claude with tools available.

**Citation Validator** (`agent/citations.py`): Parses `[file:line_start-line_end]` markers from answer text, validates each against storage, strips invalid with footnote. Pure function: `validate_citations(text, db) -> str`.

**Streamlit App** (`app.py`): ~80 lines. Session state for message history. Calls agent as a Python function. Renders citations as `st.expander`.

**Eval Runner** (`evals/run.py`): Loads `questions.jsonl`, runs each question through the agent, scores 0/1/2 by checking expected file path in citations and string-overlap with ground truth answer. Writes `results.md`.

### Interfaces

- `Chunk`: `dataclass` with `id`, `file_path`, `symbol_name`, `symbol_type`, `line_start`, `line_end`, `docstring`, `text` (full source of the symbol).
- `search_corpus(query: str, k: int = 5) -> list[Chunk]`
- `find_symbol(name: str) -> Chunk | None`
- `read_file(path: str, line_start: int, line_end: int) -> str`
- `validate_citations(text: str, db: DB) -> str`

### Architectural Decisions

- **Single process**: Agent, storage, and UI share the same Python process. No HTTP layer.
- **SQLite + sqlite-vec**: One `.db` file. Zero infrastructure. Swap to pgvector in DayOne v1 without touching the retrieval interface.
- **BM25 in-memory**: Rebuilt each run from the SQLite chunk table. Corpus (~50k LOC, ~2k chunks) is small enough that startup is under 1 second.
- **Hardcoded model**: `claude-sonnet-4-6`. No model picker — that's DayOne v1.
- **Pinned commit**: `langchain-ai/langchain` cloned at a specific commit SHA stored in `indexer/corpus_config.py`. Reproducibility over freshness.

---

## Testing Decisions

**Good test**: Tests the external behavior of a module (what it returns given inputs), not its implementation (which internal functions it calls). Tests are fast and don't require the full LangGraph graph to run.

**Modules to test:**

- **AST Chunker**: Given a `.py` file string, assert correct number of chunks emitted, correct metadata fields, correct line ranges, docstring extraction.
- **RRF Fusion**: Given two ranked lists with known scores, assert output order matches expected fused ranking. Pure function, no I/O.
- **Citation Validator**: Given answer text with valid and invalid citation markers and a stub DB, assert valid citations preserved, invalid stripped, footnote present.
- **`find_symbol`**: Given a symbol name that exists and one that doesn't, assert correct return behavior against an in-memory SQLite DB.
- **`search_corpus`**: Integration test against a small real index (5–10 chunks), assert top result is correct for known queries.

**Not tested at unit level**: Streamlit UI, LangGraph graph wiring, OpenAI embedding calls (mocked at the embedder boundary).

---

## Out of Scope

See `draft.md` §2 for full rationale. Summary:

| Cut | Deferred to |
|---|---|
| Symbol graph / DuckDB | DayOne v1, week 1 |
| `find_callers` / `find_callees` | DayOne v1, week 2 |
| Cross-package call tracing | DayOne v1, week 3 |
| Reranker | DayOne v1, week 2 |
| 6-node LangGraph | DayOne v1, week 2 |
| React/TypeScript frontend | DayOne v1, week 5 |
| FastAPI + SSE streaming | DayOne v1, week 3 |
| Multi-provider LLM picker | DayOne v1, week 3 |
| Langfuse / observability | DayOne v1, week 3 |
| Docker / hosting / landing page | DayOne v1, week 6 |
| 80-question scraped eval set | DayOne v1, week 2 |
| Blog post | DayOne v1, week 6 |

---

## Further Notes

- **Cost**: ~$0.50 OpenAI embedding cost for the full `langchain-core` corpus. Budget ~$5 total for repeated eval runs and agent queries during development.
- **Retrospective**: Day 7 includes a private retrospective answering what was harder/easier than expected and what makes you most worried about DayOne v1. This reshapes the 6-week plan before it starts.
- **Half the code carries forward**: AST chunker, retrieval pipeline, and eval scoring code transfer to DayOne v1 directly. Graph and storage get replaced.
- **Scope enforcement**: If the thought "just add the symbol graph to the MVP" appears, stop. That's the trap that turns a 7-day spike into a 3-week half-product.
