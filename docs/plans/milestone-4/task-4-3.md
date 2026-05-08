# Task 4.3 — Eval Runner

## Goal

Build `evals/run.py` that runs all 34 questions through the agent N times each, scores each run 0/1/2, reports median + variance across runs, and writes a timestamped results file to `evals/results/`.

## Acceptance Criteria

- [x] `make eval` completes without error.
- [x] Results written to `evals/results/results-<timestamp>-<agent>-<judge>.md` with per-question rows and a summary line.
- [x] Scoring is hybrid: auto file-path check (free) + LLM-as-judge (Claude Sonnet 4.6).
- [x] `tests/test_eval_runner.py` passes with mocked agent and mocked LLM judge.
- [x] N-run per question (default n=3) with median score and variance reported.
- [x] Separate agent_cost and judge_cost columns with real token counts from `usage_metadata`.
- [x] Tool trace (round, tool name, args) logged per run in results detail section.
- [x] Budget exhaustion tracked as distinct failure class (`fail/exhausted` vs `fail`).
- [x] KeyboardInterrupt writes partial results and exits cleanly.
- [x] Negative-tier questions show `n/a` for `file_ok%`; `max_total` accounts for their lower ceiling.

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

**N-run scoring**: each question runs N times (default 3). Reports `median_score`, `variance`, `file_ok%` (% of runs where file path found), `judge%` (% of runs where judge passes). Single-run scores are unreliable due to LLM non-determinism; median over 3 is the canonical score.

**Cost tracking**: per-question cost from real `usage_metadata` (both agent and judge). Agent accumulates tokens across all tool-use rounds inside `answer_node`. Separate `agent_cost` and `judge_cost` columns. Pricing lookup keyed by model prefix (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`). Models read from env vars `AGENT_MODEL` / `JUDGE_MODEL`.

**Judge failure classes**: `pass` / `fail` / `fail/exhausted` (tool budget hit before answer was complete) / `error` (exception during judge call).

**results file format** (`evals/results/results-<mmdd-hhmm>-<agent-slug>-<judge-slug>.md`):

```markdown
| id | question | median | var | file_ok% | judge% | tier | agent_cost | judge_cost |
|---|---|---|---|---|---|---|---|---|
| q01 | Where is RunnableSequence defined? | 2 | 0.00 | 100% | 100% | recall | $0.0079 | $0.0053 |
...

Median total: X / Y — Agent: $A  Judge: $B  Total: $C

---

### q01 — Where is RunnableSequence defined?

**Run 1**: score=2 file_ok=True judge=pass agent=$0.0079 judge=$0.0053
  r1: get_class_outline(class_name='RunnableSequence')
  r2: read_file(path='runnables/base.py', line_start=2875, line_end=2950)

<full agent answer>
```

Results written after each question (interrupt-safe). `KeyboardInterrupt` writes partial results and exits.

**Agent tool loop** (`answer_node.py`): `MAX_TOOL_ROUNDS = 8`. Tools: `get_class_outline(class_name)` (returns all method signatures + line ranges for a class — batched outline before any `read_file` calls) and `read_file(path, line_start, line_end)`. Temperature=0. Exhausts → forced final answer + `budget_exhausted=True` flag in `additional_kwargs`.

## Files

- `evals/run.py` — runner script
- `evals/judge_prompt.py` — `JUDGE_SYSTEM` prompt string (includes rule for negative-tier questions)
- `agent/answer_node.py` — agent with `get_class_outline` + `read_file` tools, 8 tool rounds, `budget_exhausted` flag
- `tests/test_eval_runner.py` — unit tests with mocked agent + judge

## Steps

- [x] Write `evals/run.py`: load jsonl, invoke agent per question, score, write results.
- [x] Write `tests/test_eval_runner.py`: mock agent output and judge response, assert score calculation and results format.
- [x] Run `make eval` end-to-end and verify results written correctly.
- [x] Add retry logic to `_judge` and per-question error handling.
- [x] Write results incrementally after each question (interrupt-safe).
- [x] Track per-question cost (agent + judge) using model pricing dict and real `usage_metadata`.
- [x] Fix KeyboardInterrupt (catch `BaseException` at outer loop, not `Exception` inside `_run_once`).
- [x] Pin `temperature=0` on agent and judge; log model names + temperature in run header.
- [x] N-run per question: median + variance; separate `agent_cost` and `judge_cost` columns.
- [x] Tool trace per run: `(round, tool_name, args)` from `additional_kwargs["tool_trace"]`.
- [x] `get_class_outline` tool in agent: DB.class_outline() returns class + all method chunks; outlines batched before reads.
- [x] Timestamped results filenames in `evals/results/`; `--results-dir` CLI arg.
- [x] Budget exhaustion tracking: `fail/exhausted` status; `budget_exhausted` flag in AIMessage `additional_kwargs`.
- [x] Judge rule 6: negative questions (`expected_file_paths: []`) graded on refusal quality, not corpus citation.
- [x] `file_ok%` shows `n/a` for negative tier; `max_total` adjusted accordingly.
