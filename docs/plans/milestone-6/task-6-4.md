# Task 6.4 — Question Set: Held-Out, Tier Balance, Adversarial

## Goal

Remove the strongest objection to the 94% score: the system prompt was tuned against the same 34 questions that produce the score. Also fix a tier distribution where 4 of the 7 tiers hold 1 question each.

## Acceptance Criteria

- [x] `evals/questions-test.jsonl` holds 15 or more questions that nobody used during prompt tuning.
- [x] A commit freezes the system prompt before the first test question is written. The commit SHA is recorded.
- [x] Every tier holds 5 or more questions, or `README.md` stops calling the set "7 tiers".
- [x] The negative tier holds 5 or more adversarial questions.
- [x] `make eval` accepts `--questions` and reports dev and test separately.
- [x] `README.md` describes both sets, and reports the held-out **retrieval** score.
- [ ] `README.md` reports the held-out **end-to-end** score. Blocked on A1 — `make eval-test` spends API budget.
- [x] `make check` passes.

## Problem

### Prompt tuning contaminates the score

`agent/answer_node.py:63-70` names langchain-core classes:

```
"Async sibling: for Base* classes drop the 'Base' prefix to get the async name
(e.g. BaseCallbackHandler → AsyncCallbackHandler, BaseRunManager → AsyncRunManager).
ALWAYS call get_class_outline on the async sibling"
```

That text exists to make specific eval questions pass. The 94% measures the prompt against the set that shaped it. The number is not wrong, but it does not predict behavior on a new question.

Task 6.5 removes the hack. This task builds the set that shows whether the removal costs anything.

### Tier distribution

Measured counts across the 34 questions:

| Tier | Count |
|---|---|
| behavior | 11 |
| hard | 10 |
| recall | 9 |
| definition | 1 |
| usage | 1 |
| cross-file | 1 |
| negative | 1 |

Four tiers hold a single question. A single question is a sample, not a tier. Per-tier scores for those four carry no information.

The negative tier matters most. Refusal on out-of-corpus questions is the most interesting claim in the README, and one question tests it.

## Design

### Freeze then write

1. Commit the current `agent/answer_node.py` and record the SHA in this task file.
2. Write `evals/questions-test.jsonl` from the langchain-core source only. Do not run the agent while writing. Do not adjust a question because the agent fails it.
3. Run the test set once, at the end. Report the score whatever it is.

A test set that gets edited after a failing run becomes a second dev set.

### Test set: 15 or more questions

Cover the same 7 tiers. Draw from subsystems that the dev set touches lightly. Check the `subsystem` field in `questions.jsonl` for the gaps.

### Tier balance in the dev set

Raise `definition`, `usage`, and `cross-file` to 5 or more each. These are additions to the dev set, so they may be written with the agent's output in view.

### Adversarial negative tier

Add 4 or more negative questions. Three failure shapes:

| Shape | Example |
|---|---|
| Plausible but nonexistent symbol | "What does `RunnableParallelStream` do?" |
| Real symbol, wrong package | "Where is `ChatOpenAI` defined?" |
| Real concept, not in `langchain-core` | "How does the Chroma vector store persist to disk?" |

A correct answer states that the symbol is not in the corpus. A wrong answer invents a file path. The judge already grades this shape. See rule 6 in `evals/judge_prompt.py`.

### Runner change

`run()` already takes `questions_path`. The `__main__` block already exposes `--questions`. Add a `Makefile` target for the test set, and put the question-set name in the results filename.

## Files

- `evals/questions-test.jsonl` — new
- `evals/questions.jsonl` — add dev questions for the thin tiers and the negative tier
- `evals/run.py` — put the question-set name into the results filename
- `Makefile` — add `eval-test`
- `README.md` — report the dev score and the test score

## Steps

- [x] Commit the current system prompt. Record the SHA here: `0061b3b`.
- [x] Add 4 or more `definition` questions to `evals/questions.jsonl`.
- [x] Add 4 or more `usage` questions to `evals/questions.jsonl`.
- [x] Add 4 or more `cross-file` questions to `evals/questions.jsonl`.
- [x] Add 4 or more adversarial `negative` questions across the 3 failure shapes.
- [x] Add `expected_symbols` to every new non-negative question. See task 6.3.
- [x] Write 15 or more questions in `evals/questions-test.jsonl` from the source only.
- [x] Add the question-set name to the results filename in `evals/run.py`.
- [x] Add `eval-test` to the `Makefile`.
- [x] Run both sets. Report both scores in `README.md`.
- [x] Run `make check` and confirm all tests pass.

## Result

**Freeze SHA `0061b3b`** — the last commit touching `agent/answer_node.py`, dated before milestone 6 started. Verified unchanged at the time the held-out questions were written.

**Dev set: 34 to 50 questions.** Every tier now holds 5 or more. Previously definition, usage, cross-file, and negative held exactly 1 each.

| tier | before | after |
|---|---|---|
| behavior | 11 | 11 |
| hard | 10 | 10 |
| recall | 9 | 9 |
| definition | 1 | 5 |
| usage | 1 | 5 |
| cross-file | 1 | 5 |
| negative | 1 | 5 |

New questions draw on subsystems the dev set never touched: caches, rate limiters, stores, exceptions, globals, example selectors, indexing, and serialization. The 4 new negative questions cover all 3 adversarial shapes — a plausible non-existent symbol, a real symbol in the wrong package, and a real concept outside the corpus.

**Held-out set: 17 questions**, `evals/questions-test.jsonl`, written from source only, after the freeze. 15 are graded by the retrieval eval. 7 name their gold symbol and 8 do not — near-even, against the dev set's 31 of 33. Every gold label was verified to resolve to a real chunk at an expected path.

**The held-out set immediately earned its keep.** It reversed two of the three conclusions from task 6.3 within an hour of existing. See the *Held-out correction* section of `task-6-3.md` and the reopened B1 and B1b in `open-questions.md`.

**Not run:** the end-to-end held-out score (`make eval-test`). That spends API budget, which is the owner's call — item A1 in `open-questions.md`. The retrieval score is free and was run.
