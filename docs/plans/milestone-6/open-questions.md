# Milestone 6 — Open Questions

Decisions that this milestone does not make. Each item is either blocked on a resource that only the owner controls, blocked on data that tasks 6.3 and 6.4 produce, or a design question that the audit raised and milestone 6 deliberately leaves alone.

**Discuss these after tasks 6.1 to 6.7 land.** Do not act on them during the milestone. Several depend on numbers that do not exist yet.

Status key: **OPEN** — needs a decision. **BLOCKED** — needs data first. **PARKED** — a known defect that milestone 6 chose not to fix.

---

## A. Needs the owner's budget or time

### A1. Re-run the eval at n=3 — OPEN

`README.md` publishes 63/67 at n=1. The runner defaults to n=3 and reports a median and a variance. A single sample carries no variance figure, so the headline number is the weakest form of the eval that the code supports.

Cost: 34 questions, 3 runs, one agent call and one judge call per run. Task 6.1 corrected the price table, so the figure the runner prints is now real.

Decision: spend the budget, or keep the n=1 number with its stated caveat. Task 6.2 left the caveat in place.

### A2. Who writes the labels and the held-out questions — OPEN

Task 6.3 needs `expected_symbols` on 33 questions. Task 6.4 needs 15 or more held-out questions plus roughly 12 new dev questions.

The labels must come from the langchain-core source, not from the current retrieval output. A label read off the output measures nothing. That makes the work slow and hard to delegate to the agent that the labels grade.

Decision: who writes them, and whether an agent may draft them for human review.

### A3. Judge validation sample — OPEN

Task 6.7 needs 40 hand-labeled `(question, answer)` pairs to compute agreement with the LLM judge. The labeler must not see the judge verdict first.

Same constraint as A2: the point of the exercise is an independent human opinion.

---

## B. Blocked on data from tasks 6.3 and 6.4

### B1. Keep or delete the symbol pre-check — ANSWERED by 6.3: keep it

The ablation is decisive. The pre-check lifts recall@5 from 63.6% to 97.0%, and MRR from 0.439 to 0.924. It is the highest-value component in the retriever, not the implementation detail the README treated it as.

Two follow-ups it raises instead:

1. **RRF is the weak link, not the pre-check.** Hybrid fusion scores below dense-only at k=5 (63.6% against 69.7%). Dropping BM25 and running dense plus the pre-check is now a live option, and it would delete the BM25 index, its 380 ms cold start, and the tokenizer. Test that fifth configuration before deciding.
2. **The 97% is measured on a favorable set.** 31 of 33 questions name their gold symbol verbatim, and the pre-check is exact name lookup. Re-run the ablation against the task 6.4 held-out questions before quoting the number anywhere.

### B2. Reranker — ANSWERED by 6.3: keep it out of scope

The recall@5 to recall@10 gap bounds what reranking could recover. With the pre-check in place there is no gap at all: 97.0% at both cutoffs. Every chunk a reranker could promote is already in the top 5.

The gap is large without the pre-check (RRF hybrid: 63.6% to 81.8%, so 18 points sit in ranks 6 to 10). That is the measured argument for keeping the pre-check, not for adding a reranker.

Revisit only if B1 follow-up 2 shows the pre-check collapsing on held-out questions.

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
