# ast-rag Domain Glossary

## Chunk
Unit of indexing and retrieval. One `Chunk` per top-level function, top-level class, or method defined on a class. Methods are sibling chunks — not nested under the class chunk. Each carries: `file_path`, `symbol_name`, `symbol_type` (`"function"` | `"class"` | `"method"`), `parent_class` (for methods, else `None`), `line_start`, `line_end`, `docstring`.

**Embed text**: for methods, text is prefixed `"{parent_class}.{symbol_name}: {raw_source}"` before embedding and BM25 indexing. Raw source stored separately for `read_file` display. Class context injected at index time, not stored in the chunk text field.

## Chunk Hash
Idempotency key for embedder: `sha256(f"{file_path}{symbol_name}{line_start}{line_end}{text}")`. Stored in Index alongside embedding. Embedder skips chunk if hash already present. Re-embeds only when chunk content actually changes (e.g. new pinned commit).

## Corpus
The `langchain-core` package source, cloned at a pinned commit SHA. Fixed and read-only during a session. Unit of indexing.

## Index
The SQLite `.db` file containing: chunk metadata table, sqlite-vec embeddings table, BM25 text corpus (in-memory at runtime). Single artifact produced by the Indexer.

## Citation Marker
`[runnables/base.py:120-180]` format — short path (corpus root stripped) + line range. Embedded in agent answer text. Validated against Index before return; invalid markers stripped with footnote. One `CORPUS_ROOT` constant handles prefix stripping at validation time.

## Citation Expander
Streamlit `st.expander` for each citation. Shows: (1) raw source lines in monospace code block; (2) GitHub permalink `https://github.com/langchain-ai/langchain/blob/{COMMIT_SHA}/libs/core/{path}#L{start}-L{end}` using pinned commit SHA. SHA stored in `indexer/corpus_config.py`.

## Symbol Lookup
`find_symbol(name)` matches case-insensitively on `symbol_name`. Returns first match or `None`. Not fuzzy, not prefix — exact modulo case.

## read_file Bounds
Max 100 lines per call (`end - start <= 100`). If exceeded, clamps to 100 and appends `"[truncated: requested N lines, returned 100]"` to result. Claude can re-call with narrower range.

## BM25 Tokenization
Whitespace split + camelCase expansion + snake_case expansion. `"RunnableSequence"` → `["Runnable", "Sequence", "RunnableSequence"]`. `"invoke_async"` → `["invoke", "async", "invoke_async"]`. Applied symmetrically to both chunk text at index time and query text at search time. ~10 lines regex, no external tokenizer.

## RRF (Reciprocal Rank Fusion)
Merge strategy for BM25 top-10 and dense top-10 results. Produces a unified top-5 without learned weights. No reranker. Formula: `score(d) = Σ 1 / (k + rank(d))`, `k=60` (standard, hardcoded). Chunks appearing in only one list are still included.

## System Prompt Citation Rule
Hard requirement in system prompt: "You MUST cite every factual claim with `[path:start-end]`. Never state a fact without a citation. If retrieved chunks do not support a claim, say 'I don't have source for this' instead of stating it uncited." Escape valve prevents hallucinated markers; hard requirement drives citation recall in eval.

## Retrieve Node Logic
Heuristic pre-check: regex extracts CamelCase / `snake_case` / `ALL_CAPS` tokens from query → checks against symbol name set (loaded at startup from Index). If match → `find_symbol` first; merge result into `retrieved_chunks`, then `search_corpus` for remaining slots. If no match → `search_corpus` only. Zero extra LLM calls.

## Agent State
LangGraph `TypedDict` with two fields: `messages: list[BaseMessage]` (LangChain message history) and `retrieved_chunks: list[Chunk]`. `retrieve` node writes chunks once (replace reducer); `answer` node reads them. Explicit field — not inferred from message history — so citation validator and eval runner can inspect chunks directly.

## Answer Node
Runs a tool-call loop (`MAX_TOOL_ROUNDS = 8`). Each round: invoke LLM (model from `AGENT_MODEL` env var, default `claude-haiku-4-5`) → if `AIMessage` contains tool calls → execute tools → feed `ToolMessage` back → repeat. Stops when model returns a plain text response or 8 rounds are exhausted. Budget exhausted → forced final answer + `budget_exhausted=True` in `AIMessage.additional_kwargs`. Tools: `get_class_outline(class_name)` (all method signatures + line ranges for a class — agent batches async sibling + all mixins before any reads) and `read_file(path, line_start, line_end)`. Retrieved chunks injected into system prompt alongside citation rule; user message is the raw query. Tool trace `(round, tool_name, args)` accumulated in `additional_kwargs["tool_trace"]`.

## Citation Validator
Parses `[path:start-end]` markers from answer text. Validates each via `db.chunk_exists_at(path, start, end)`. Strips invalid markers; appends `"*N citation(s) could not be verified and were removed.*"` footnote at end if any were stripped.

## Eval
34 hand-crafted questions scored 0/1/2. Hybrid scoring: (1) auto-check any path in `expected_file_paths` appears in answer (objective, free); (2) LLM-as-judge (Claude Sonnet 4.6) rates answer quality against `description_must_include` / `description_must_not_assert` (subjective). N-run per question (default n=3); canonical score = median across runs. Results written to `evals/results/results-<mmdd-hhmm>-<agent>-<judge>.md` with per-question rows (median, variance, file_ok%, judge%, agent_cost, judge_cost) and per-run tool traces. Baseline: 63/67 (94%) haiku-4-5 agent, n=1.

**Score rubric**: 2 = file_ok AND judge pass; 1 = one of the two passes; 0 = neither. Negative-tier questions max = 1 (no file_ok check). Judge failure classes: `pass` / `fail` / `fail/exhausted` (budget hit) / `error`.

**Question schema** (`evals/questions.jsonl`): `id`, `question`, `expected_file_paths` (list — empty `[]` for negative tier), `description_must_include`, `description_must_not_assert`, `tier` (`recall` | `behavior` | `hard` | `definition` | `usage` | `cross-file` | `negative`), `subsystem`. Lines with `_meta` key skipped by runner. Negative tier: judge checks refusal quality (model must say answer is not in corpus).
