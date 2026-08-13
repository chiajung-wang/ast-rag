# ast-rag Domain Glossary

## Chunk
Unit of indexing and retrieval. One `Chunk` per top-level function, top-level class, or method defined on a class. Methods are sibling chunks — not nested under the class chunk. Each carries: `file_path`, `symbol_name`, `symbol_type` (`"function"` | `"class"` | `"method"`), `parent_class` (for methods, else `None`), `line_start`, `line_end`, `docstring`, `base_classes` (for classes, else empty).

**Embed text**: for methods, text is prefixed `"{parent_class}.{symbol_name}: {raw_source}"` before embedding and BM25 indexing. Raw source stored separately for `read_file` display. Class context injected at index time, not stored in the chunk text field.

## Chunk Hash
Idempotency key for embedder: `sha256(f"{file_path}{symbol_name}{line_start}{line_end}{text}")`. Stored in Index alongside embedding. Embedder skips chunk if hash already present. Re-embeds only when chunk content actually changes (e.g. new pinned commit).

## Corpus
The `langchain-core` package source, cloned at a pinned commit SHA. Fixed and read-only during a session. Unit of indexing.

## Index
The SQLite `.db` file containing: chunk metadata table, sqlite-vec embeddings table, BM25 text corpus (in-memory at runtime). Single artifact produced by the Indexer.

## Citation Marker
`[runnables/base.py:120-180]` format — short path (corpus root stripped) + line range. Embedded in agent answer text. Checked against Index before return. Markers that fail are stripped with a footnote.

**Path normalization**: `normalize_path` in `agent/citations.py` strips any leading corpus-root segment before the lookup, so `langchain_core/runnables/base.py` and `libs/core/langchain_core/runnables/base.py` both resolve. A marker that survives is rewritten to the canonical short path, so every consumer downstream sees one form. Prefixes derive from `CLONE_DIR` and `CORPUS_SUBPATH`.

**What the check proves**: the cited range falls inside an indexed symbol in that file. `chunk_exists_at` tests containment (`line_start <= ? AND line_end >= ?`). It does not prove the lines support the claim.

## Citation Expander
Streamlit `st.expander` for each citation. Shows: (1) raw source lines in monospace code block; (2) GitHub permalink `{REPO_URL}/blob/{COMMIT_SHA}/{CORPUS_SUBPATH}/{path}#L{start}-L{end}`.

The prefix comes from `CORPUS_SUBPATH`, never a literal. `file_path` is relative to `CLONE_DIR/CORPUS_SUBPATH`, so a hardcoded `libs/core/` drops the `langchain_core/` segment and every link returns 404.

## Symbol Lookup
`find_symbol(name)` matches case-insensitively on `symbol_name`. Returns one match or `None`. Not fuzzy, not prefix — exact modulo case.

**Tie-break**: 1296 distinct names cover 2414 chunks, so a name can match many chunks (`__init__` matches 108, `invoke` matches 23). The query orders by `symbol_type` (class, then method, then function), then by `file_path`, then by `line_start`, and takes the first row. The rule is arbitrary but stable. Without it SQLite returns an arbitrary row, and the retrieve node injects that chunk ahead of the RRF results.

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
Runs a tool-call loop (`MAX_TOOL_ROUNDS = 8`). Each round: invoke LLM (model from `AGENT_MODEL`, default `anthropic/claude-haiku-4.5`, via OpenRouter) → if `AIMessage` contains tool calls → execute tools → feed `ToolMessage` back → repeat. Stops when model returns a plain text response or 8 rounds are exhausted. Budget exhausted → forced final answer + `budget_exhausted=True` in `AIMessage.additional_kwargs`. Tools: `get_class_outline(class_name)` (own + inherited method signatures, external bases, direct subclasses — one call maps a class) and `read_file(path, line_start, line_end)`. Retrieved chunks injected into system prompt alongside citation rule; user message is the raw query. Tool trace `(round, tool_name, args)` accumulated in `additional_kwargs["tool_trace"]`.

## Citation Validator
Parses `[path:start-end]` markers from answer text. Validates each via `db.chunk_exists_at(path, start, end)`. Strips invalid markers; appends `"*N citation(s) could not be verified and were removed.*"` footnote at end if any were stripped.

## Eval
Two sets: dev (`evals/questions.jsonl`, 50 questions, prompt tuned against the original 34) and held-out (`evals/questions-test.jsonl`, 17 questions written from source after the prompt froze at `0061b3b`). Scored 0/1/2. Hybrid scoring: (1) auto-check any path in `expected_file_paths` appears in answer (objective, free); (2) LLM-as-judge (Claude Sonnet 4.6) rates answer quality against `description_must_include` / `description_must_not_assert` (subjective). N-run per question (default n=3); canonical score = median across runs. Results written to `evals/results/results-<mmdd-hhmm>-<agent>-<judge>.md` with per-question rows (median, variance, file_ok%, judge%, agent_cost, judge_cost) and per-run tool traces. Baseline: 63/67 (94%) haiku-4-5 agent, n=1, original 34 dev questions. Held-out end-to-end score not yet run.

**Score rubric**: 2 = file_ok AND judge pass; 1 = one of the two passes; 0 = neither. Negative-tier questions max = 1 (no file_ok check). Judge failure classes: `pass` / `fail` / `fail/exhausted` (budget hit) / `error`.

**Question schema**: `id`, `question`, `expected_file_paths` (list — empty `[]` for negative tier), `expected_symbols` (gold symbols for the retrieval eval), `description_must_include`, `description_must_not_assert`, `tier` (`recall` | `behavior` | `hard` | `definition` | `usage` | `cross-file` | `negative`), `subsystem`. Lines with `_meta` key skipped by runner. Negative tier: judge checks refusal quality (model must say answer is not in corpus).

## Class Outline
`db.class_outline(name)` returns a `ClassOutline`: the class chunk, `OutlineEntry` rows for its own and inherited methods, `external_bases` it could not resolve, and `subclasses` defined in the corpus.

**Why inheritance matters here**: `BaseCallbackHandler` defines only 7 `ignore_*` flags. Every `on_*` event lives on one of 6 mixins it inherits, and the async variants live on `AsyncCallbackHandler`, a subclass. Before task 6.5 the outline returned 8 rows and no events, and the system prompt compensated with a hardcoded rule naming langchain-core classes. It now returns 29 rows including 20 events.

**Walk**: breadth-first from the class, capped at `MAX_MRO_DEPTH = 3`, with a visited set that tolerates inheritance cycles. The first definition of a method name wins, so a subclass override hides the base method.

**Scoping**: methods are matched by `parent_class` *and* `file_path`. Matching on name alone merged same-named classes from different modules (`NoLock`, `RunInfo`, `Tee`, `ToolCall`, `ToolCallChunk` each appear twice).

**Base names**: `indexer/chunker.py:_base_names` records simple names only. `mod.Foo` stores `Foo`, `Generic[T]` stores `Generic`. `base_classes` is excluded from the chunk hash, so `insert_chunk` upserts the column and a re-index fills it without invalidating an embedding.

## Provider
OpenRouter is the only provider. One key, `OPENROUTER_API_KEY`, covers chat and embeddings. Base URL and model defaults live in `provider.py`.

Chat uses `ChatOpenAI` and embeddings use `OpenAI`, both pointed at `https://openrouter.ai/api/v1`. There is no Anthropic SDK dependency. Provider failures surface as `openai.APIError`.

**Model slugs** use dots, not dashes: `anthropic/claude-haiku-4.5`. A wrong slug 404s at request time and also misses the eval price table, which silently falls back to the Sonnet rate.

## Embed Model Guard
Embeddings from two models do not share a vector space. A query embedded by model A against an index built by model B returns plausible, wrong neighbours and raises nothing.

`indexer/embedder.py` writes the active `EMBED_MODEL` into a `meta` table at index time. `retrieval/pipeline.py` calls `db.assert_embed_model()` when it first opens the index, raising `EmbedModelMismatch` on disagreement. An index built before the guard existed records no model and is allowed through, so upgrading does not break an existing `.db`.
