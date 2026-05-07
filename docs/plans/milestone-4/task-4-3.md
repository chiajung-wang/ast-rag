# Task 4.3 — Eval Runner

## Goal

Build `evals/run.py` that runs all 30 questions through the agent, scores each 0/1/2, and writes `evals/results.md`.

## Acceptance Criteria

- [x] `make eval` completes without error.
- [x] `evals/results.md` written with per-question rows and a summary line.
- [x] Scoring is hybrid: auto file-path check (free) + LLM-as-judge (Claude Sonnet 4.6).
- [x] `tests/test_eval_runner.py` passes with mocked agent and mocked LLM judge.

## Design

**Scoring rubric** (per question, max 2 points):

| Score | Condition |
|---|---|
| 2 | `file_ok = True` AND `judge = pass` |
| 1 | `file_ok = True` XOR `judge = pass` |
| 0 | `file_ok = False` AND `judge = fail` |

- `file_ok`: any path in `expected_file_paths` appears in the answer.
- `judge`: LLM-as-judge (Claude Sonnet 4.6) given structured JSON payload → returns JSON verdict; `description_correct` field used as pass/fail.

**LLM judge prompt**: defined in `evals/judge_prompt.py` (`JUDGE_SYSTEM`). Judge receives `{question, expected_file_paths, description_must_include, description_must_not_assert, model_answer}` and returns `{path_correct, description_correct, overall_correct, ...}`.

**Retry**: `_judge` retries up to 3 times with exponential backoff (1s, 2s). Per-question try/except logs errors as `score=0 judge=error` without aborting the run.

**Cost tracking**: per-question cost computed from actual `usage_metadata` (judge) and estimated token count from message lengths (agent). Pricing lookup keyed by model prefix (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`). Models read from env vars `AGENT_MODEL` / `JUDGE_MODEL`.

**results.md format**:

```markdown
| id | question | score | file_ok | judge | tier | cost |
|---|---|---|---|---|---|---|
| q01 | Where is RunnableSequence defined? | 2 | ✓ | pass | recall | $0.0023 |
...

Total: X / 60 — Cost: $Y.YYYY

---

### q01 — Where is RunnableSequence defined? ($0.0023)

<full agent answer>
```

Results written after each question (interrupt-safe).

## Files

- `evals/run.py` — runner script
- `evals/judge_prompt.py` — `JUDGE_SYSTEM` prompt string
- `tests/test_eval_runner.py` — unit tests with mocked agent + judge

## Steps

- [x] Write `evals/run.py`: load jsonl, invoke agent per question, score, write results.md.
- [x] Write `tests/test_eval_runner.py`: mock agent output and judge response, assert score calculation and results.md format.
- [x] Run `make eval` end-to-end and verify `evals/results.md` written correctly.
- [x] Add retry logic to `_judge` and per-question error handling.
- [x] Write results incrementally after each question (interrupt-safe).
- [x] Track per-question cost (agent + judge) using model pricing dict and usage metadata.
