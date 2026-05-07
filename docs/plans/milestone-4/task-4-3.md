# Task 4.3 — Eval Runner

## Goal

Build `evals/run.py` that runs all 20 questions through the agent, scores each 0/1/2, and writes `evals/results.md`.

## Acceptance Criteria

- [ ] `make eval` completes without error.
- [ ] `evals/results.md` written with per-question rows and a summary line.
- [ ] Scoring is hybrid: auto file-path check (free) + LLM-as-judge (Claude Sonnet 4.6).
- [ ] `tests/test_eval_runner.py` passes with mocked agent and mocked LLM judge.

## Design

**Scoring rubric** (per question, max 2 points):

| Score | Condition |
|---|---|
| 2 | `file_ok = True` AND `judge = pass` |
| 1 | `file_ok = True` XOR `judge = pass` |
| 0 | `file_ok = False` AND `judge = fail` |

- `file_ok`: `expected_file_path` appears in citation markers of the answer.
- `judge`: LLM-as-judge (Claude Sonnet 4.6) given `question`, `ground_truth_answer`, `agent_answer` → returns `pass` or `fail`.

**LLM judge prompt** (system): "You are an evaluation judge. Given a question, a ground truth answer, and an agent answer, respond with exactly 'pass' if the agent answer is substantially correct, or 'fail' otherwise. No explanation."

**results.md format**:

```markdown
| id | question | score | file_ok | judge |
|---|---|---|---|---|
| q01 | Where is RunnableSequence defined? | 2 | ✓ | pass |
...

Total: X / 40
```

## Files

- `evals/run.py` — runner script
- `tests/test_eval_runner.py` — unit tests with mocked agent + judge

## Steps

- [ ] Write `evals/run.py`: load jsonl, invoke agent per question, score, write results.md.
- [ ] Write `tests/test_eval_runner.py`: mock agent output and judge response, assert score calculation and results.md format.
- [ ] Run `make eval` end-to-end and verify `evals/results.md` written correctly.
