# Milestone 6 — Open Questions

Decisions that this milestone does not make. Each item is either blocked on a resource that only the owner controls, blocked on data that tasks 6.3 and 6.4 produce, or a design question that the audit raised and milestone 6 deliberately leaves alone.

**Discuss these after tasks 6.1 to 6.7 land.** Do not act on them during the milestone. Several depend on numbers that do not exist yet.

Status key: **OPEN** — needs a decision. **BLOCKED** — needs data first. **PARKED** — a known defect that milestone 6 chose not to fix.

---

## A. Needs the owner's budget or time

> **The Anthropic account has no credit.** Verified 2026-08-12: any agent or judge call returns `400 invalid_request_error — Your credit balance is too low to access the Anthropic API`. Every item in this group that spends Anthropic tokens (A1, A3, A4) is blocked on topping up, not on deciding to spend. Measured cost is small: ~$0.62 for a held-out n=1 run, ~$1.82 for dev n=1, ~$5.46 for dev n=3. OpenAI is unaffected, so `make eval-retrieval` still runs.

### A1. Re-run the eval at n=3 — PARTIALLY CLOSED, one open decision

Run 2026-08-12, dev set (50 questions, 45×2 + 5 negative×1 = 95 max), `anthropic/claude-sonnet-5` agent+judge via OpenRouter, n=3: **91.0 / 95 (95.8%)**, $9.96 ($8.74 agent + $1.21 judge). Replaces the old 63/67 (94%, n=1, haiku-4-5, direct Anthropic, 34 questions) — not a fair comparison to begin with, since provider, model, and question count all changed.

Investigating the 4 non-perfect rows found two unrelated bugs, both now fixed and independently re-verified (see the `fix(agent)` and `fix(eval)` commits, 2026-08-13):

- **q20** — not a grading issue. Run 2 was a silent crash: `score=0 agent_cost=$0.00`, no message. `agent/answer_node.py` only caught `openai.APIError`; a `json.JSONDecodeError` from the SDK's own response parsing escaped uncaught. Fixed by broadening the catch. This does **not** retroactively make q20 a clean pass — the underlying cause (the model occasionally emitting malformed tool-call JSON) still happens, and the fix only ensures it degrades to a graceful message instead of a $0 zombie result. q20's median=1 stands as a real, measured flakiness score.
- **q28, q29, q30** — genuine eval-authoring bugs, predating task 6.4. Each `description_must_include` demanded a fact absent from the indexed corpus entirely (grep-verified), which the system prompt's own citation rule forbids the agent from stating. Fixed in `evals/questions.jsonl`. Each independently re-confirmed 2/2 post-fix (n=1, ~$0.24 total).

**What's not done:** a clean n=3 re-run of the full 50-question set with both fixes in place. The 91.0/95 figure above still reflects the pre-fix criteria on q28/q29/q30 (each capped at 1/2 by a bug, now fixed) and the pre-fix crash handling on q20 (unrelated to its true score). A defensible read: 91.0/95 plus the +3 from q28/q29/q30's fix ≈ **94/95 (~99%)**, but that mixes an n=3 aggregate with n=1 spot-checks and isn't a number to publish as-is.

Decision: spend another ~$10 for one clean n=3 run to get a single, methodologically consistent headline number, or publish 91.0/95 with this note attached and let a future run supersede it. Either way, this now also unblocks the 6.7 regression baseline and the 6.5 accuracy check, which were gated on A1 landing at all.

### A2. Who writes the labels and the held-out questions — OPEN

Task 6.3 needs `expected_symbols` on 33 questions. Task 6.4 needs 15 or more held-out questions plus roughly 12 new dev questions.

The labels must come from the langchain-core source, not from the current retrieval output. A label read off the output measures nothing. That makes the work slow and hard to delegate to the agent that the labels grade.

Decision: who writes them, and whether an agent may draft them for human review.

### A5. Dev ablation is stale relative to the dev set — CLOSED

Refreshed 2026-08-12 via `make eval-retrieval`. `verify_embeddings.py` had already confirmed OpenRouter's `text-embedding-3-small` matches the stored index exactly, so this only needed to embed the 12 new questions — cents, no re-index.

New dev numbers (45 questions, was 33): BM25 42.2% (was 39.4%), dense 73.3% (was 69.7%), RRF hybrid 68.9% (was 63.6%), pre-check 95.6% (was 97.0%). Every configuration moved a few points and **none crossed another** — the ranking (pre-check > dense > RRF > BM25) and every conclusion in B1/B1b are unaffected. Updated in `README.md`, `CLAUDE.md`, and appended to `docs/plans/milestone-6/task-6-3.md`.

### A3. Judge validation sample — CLOSED

Human labeled all 40 rows blind (2026-08-13), sampled from the 150-run dev n=3 results file. Raw agreement 82.5%, κ=0.440 — looks bad, but 7 of the 40 rows are rubric-drift, not real disagreement: all 7 are q28/q29/q30 (the eval-authoring bug fixed while investigating A1), all `human=pass judge=fail`, where the stored "judge" verdict predates that fix and the human correctly labeled against the corrected rubric already sitting in the blind file. Excluding those:

**Clean comparison: n=33, agreement 32/33 (97.0%), κ=0.872.** Clears the ≥0.8 bar task 6.7 asks for.

Exactly one real disagreement survives: `q03-run1`, "Where is `ChatPromptTemplate` defined?" — human said fail, original judge said pass. The automated cross-judge check (haiku-4.5, run earlier the same day) independently flagged the same row for the same reason: the answer lists the class's methods and location in full but never states the rubric's required sentence — *"builds chat-style prompts from message templates"* — explicitly. Two independent evaluators, one human and one different model, agree the original judge was too willing to infer a stated fact from field names. A real, narrow, corroborated finding about judge strictness, not a systemic problem — narrow because it's 1 of 33 clean comparisons and it isn't touched here (see the note below task-6-7 about leaving this open).

Cross-judge check (haiku-4.5, fully automated, no bias concern): 34/40 raw, same rubric-drift artifact — 39/40 (97.5%) once the same 7 rows are excluded, corroborating the human result independently.

### A4. Confirm prompt caching engages inside the real tool loop — CLOSED

Confirmed 2026-08-12, agent=judge=`anthropic/claude-sonnet-5` via OpenRouter: `t11` (held-out, the one every retrieval config in 6.4 missed) reported `cache_read=33159` on its single run. Real cache hit inside a live `answer_node` call, not a standalone probe.

`format_results_md` did not render `cache_read_tokens` even though `_run_once` captured it — the earlier claim that "any `make eval` closes this as a side effect" was wrong; the field existed but was invisible in every prior results file. Fixed in the same pass: the per-run line now prints `cache_read=<n>`.

Finding this also surfaced a real bug: a first attempt at this exact question crashed with an uncaught pydantic `ValidationError` (sonnet-5 packed `path:start-end` into `read_file`'s `path` field instead of 3 separate args) — `answer_node` only caught `openai.APIError`, so the exception propagated past `graph.invoke` and the eval layer scored the question 0. Fixed: tool invocation is now wrapped per call, and a malformed arg set becomes an `is_error` `ToolMessage` fed back to the model, the same recovery path a bad path or line range already used. Verified against a repro that fails pre-fix and passes after. Tests 187 → 188.

---

## B. Blocked on data from tasks 6.3 and 6.4

### B1. Keep or delete the symbol pre-check — REOPENED by 6.4

**Task 6.3 answered "keep it, decisively". Task 6.4 withdrew that answer.** The 6.3 evidence was the dev set, where the pre-check lifts recall@5 from 63.6% to 97.0%. The held-out set says something different:

| configuration | dev recall@5 | held-out recall@5 | held-out MRR |
|---|---|---|---|
| RRF hybrid | 63.6% | 93.3% | 0.813 |
| RRF + symbol pre-check | 97.0% | 93.3% | 0.880 |

On held-out the pre-check adds **no recall at all**, and the split by phrasing is identical for both rows: 7/7 on questions that name the symbol, 7/8 on questions that do not. The dev set has 31 of 33 questions naming their own symbol; the held-out set has 7 of 15. The 97% measured symbol lookup, not retrieval.

What survives: the pre-check improves **ranking**. MRR 0.880 against 0.813 means it puts the right chunk nearer the top of the list it already had. That is worth something to an agent reading 5 chunks in order, but it is a much smaller claim than "the component that carries the retriever".

Still open, and now the more interesting question: is a ranking gain worth the code? Deciding needs a bigger held-out set. At n=15 a single question moves recall by 6.7 points.

### B1b. RRF versus dense-only — REOPENED by 6.4

Task 6.3 concluded that RRF hurts, because on dev it scored below dense-only (63.6% against 69.7%), and proposed deleting BM25.

Held-out reverses it. RRF is the best configuration there (93.3% against dense 86.7%), and BM25 alone nearly doubles from 39.4% to 73.3%. Lexical search looks weak on a set where every question already contains the exact symbol name, because dense retrieval handles those too. It earns its place once questions are phrased normally.

**Do not delete BM25.** The proposal in 6.3 rested on the biased set.

### B2. Reranker — ANSWERED: keep it out of scope

The recall@5 to recall@10 gap bounds what reranking could recover. It is zero on both sets for the shipped configuration:

| set | RRF + pre-check recall@5 | recall@10 |
|---|---|---|
| dev | 97.0% | 97.0% |
| held-out | 93.3% | 93.3% |

Every chunk a reranker could promote is already in the top 5. This is the one conclusion from 6.3 that held-out did **not** disturb, and it holds for plain RRF on held-out too (93.3% at both cutoffs).

### B3. Cost of removing the prompt hack — BLOCKED on 6.4 and 6.5

Task 6.5 deletes the langchain-core naming rules from the system prompt. Task 6.5 requires the held-out score to hold steady.

If the score drops, the outline tool is still missing something that the hack supplied. The rule is: find the gap in the tool. Do not put the class names back.

Decision if it drops: what the tool still lacks, and whether that is worth another task.

---

## C. Measured defects that milestone 6 does not fix

### C1. Class chunks duplicate their own method chunks — PARKED

A class chunk stores the whole class body. Each method is also a sibling chunk. The same source is therefore indexed twice.

Measured on the current index:

| Class | Class chunk | Method chunks | Share duplicated |
|---|---:|---:|---:|
| `Runnable` | 93,441 chars | 88,398 chars over 54 methods | 94.6% |
| `BaseChatModel` | 89,129 chars | 80,324 chars over 40 methods | 90.1% |
| `BaseLLM` | 41,576 chars | 40,989 chars over 26 methods | 98.6% |
| `VectorStore` | 34,298 chars | 33,905 chars over 35 methods | 98.9% |

Corpus totals: 331 class chunks hold 1,369,613 chars, and 1533 method chunks hold 1,161,790 chars, out of 3,269,719 chars in all.

Three consequences:

1. BM25 counts every method token twice, once in the method chunk and once in the class chunk. That shifts the IDF values that ranking depends on.
2. The embedding bill covers the same text twice.
3. RRF can return a class chunk and one of its own methods as two of the top 5 results, which spends a slot on content the other slot already carries.

`CONTEXT.md` states the sibling design on purpose, and the design is sound. The duplication is a side effect that nobody measured.

Options: store a class chunk that holds the signature, the docstring, and the method list, but not the method bodies. Or keep the current text and exclude class chunks from the BM25 corpus. Both change the index, so both need the task 6.3 metrics in place first to show the effect.

### C2. The 7 most central classes are truncated before embedding — PARKED

`indexer/embedder.py` cuts `embed_text` at `MAX_CHARS = 24_000` to stay under the 8192-token limit. Exactly 7 chunks exceed it, and they are the 7 classes a user is most likely to ask about:

| Symbol | Size | Share embedded |
|---|---:|---:|
| `Runnable` | 93,441 chars | 26% |
| `BaseChatModel` | 89,129 chars | 27% |
| `BaseLLM` | 41,576 chars | 58% |
| `VectorStore` | 34,298 chars | 70% |
| `RunnableLambda` | 30,201 chars | 79% |
| `BaseTool` | 28,626 chars | 84% |
| `RunnableSequence` | 27,739 chars | 87% |

The `Runnable` embedding represents the class header and the first few methods. It does not represent the class. Dense retrieval for a question about a late method on `Runnable` has to reach the method chunk, because the class vector cannot carry it.

The truncation itself is correct and necessary. The problem is what gets truncated, and C1 is the reason the text is that large. Fixing C1 removes most of C2.

Task 6.3 can measure the effect directly: check recall on questions that target a late method of a large class.

### C3. `find_symbol` returns one chunk and hides the rest — PARKED

Task 6.1 replaced an arbitrary row with a stable tie-break: class, then method, then function, then path, then line. The rule is deterministic and documented. It is still arbitrary.

A user who asks about `invoke` gets one of 23 chunks with that name, and nothing tells the model that the other 22 exist.

Options: return the top few candidates and let the answer node choose. Or expose a `find_symbol_all(name)` tool. Both widen the tool surface, which the milestone-5 scope boundary resisted.

### C4. Citation checking uses containment — PARKED

`chunk_exists_at` passes when the cited range sits inside any chunk for that file. A citation of lines 900-905 passes on the strength of the 891-1205 class chunk, whatever is on those lines.

Task 6.2 documented the guarantee honestly. Task 6.7 adds a citation-precision metric that measures the stronger property.

Decision after 6.7: whether the metric justifies a stricter check. A stricter check would raise the strip rate, and stripping a correct citation is worse for a reader than keeping a loose one.

### C7. Module-level assignments are never indexed — PARKED

Found while labeling gold symbols for task 6.3. `indexer/chunker.py` visits only `FunctionDef`, `AsyncFunctionDef`, and `ClassDef` at module level. Every `Assign` and `AnnAssign` is skipped, and there are **265 of them across the 181 files**.

Question q21 asks what `RunnableMap` resolves to. The answer is one line, `runnables/base.py:4151`:

```python
RunnableMap = RunnableParallel
```

That line is not in the index and cannot be retrieved. The agent can still reach it with `read_file`, so the end-to-end eval passes the question, and the retrieval eval had to be graded against `RunnableParallel` instead.

Aliases, module-level constants, `__all__`, and type aliases are all invisible to retrieval for the same reason.

Options: chunk module-level assignments whose target is a Name, or add one synthetic module chunk per file holding the assignments and the imports. Both raise the chunk count, so measure with task 6.3 metrics before and after.

### C5. `docstring` is stored and never used for retrieval — PARKED

Every chunk carries a `docstring` column. `get_class_outline` prints the first line. Nothing else reads it. It is not weighted in BM25 and not separated in the embed text.

A docstring is the highest-signal text in a chunk for a natural-language question. Worth an experiment once task 6.3 can measure a retrieval change.

### C6. Prompt caching may not clear the minimum prefix — BLOCKED on 6.6

Task 6.6 adds `cache_control` to the chunk block in the system prompt. The minimum cacheable prefix depends on the model, and Haiku 4.5 needs 4096 tokens. Five chunks usually clear that. A short retrieval may not, and the block then fails to cache with no error.

Task 6.6 reports `cache_read_input_tokens` so the answer is measured. Decide the fallback after seeing the number.

---

## D. Project-level

### D1. Corpus SHA refresh policy — OPEN

The index pins `1519ed5a`. langchain-core keeps moving. The pin gives reproducible citations and permalinks, which is the right default.

Open: when to re-pin, who checks that the eval answers still hold at a new SHA, and whether a stale pin reads as care or as neglect. Note that the eval questions encode file paths, so a re-pin can invalidate questions.

### D2. Deployment and concurrency — OPEN

`README.md` now records that a second concurrent Streamlit session raises `ProgrammingError`. Task 6.6 makes the connection thread-safe for reads.

Open: whether a hosted demo is wanted. A hosted demo needs an API key budget, rate limiting, and abuse controls. None of that is in scope today.

### D3. Which numbers to quote outside the repo — OPEN

After 6.3, 6.4, and 6.7 there will be several: end-to-end dev score, end-to-end held-out score, recall@5, MRR, the ablation delta, citation precision, and judge agreement.

The held-out score and the ablation delta are the two that survive scrutiny. The dev score alone invites the question that task 6.4 exists to answer.

Decide once the numbers exist. Do not quote a number before its task lands.

### D4. `make` targets assume an activated venv — OPEN

Every Makefile target calls bare `python`, so `make check` fails unless the caller activated `.venv` first. Task 6.6 notes it beside the CI work, because `uv run` sidesteps it in CI while a local developer still hits it.

Decision: change the targets to `uv run python`, or state the activation step in `README.md`.
