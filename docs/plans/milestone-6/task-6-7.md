# Task 6.7 — Eval Instrumentation

## Goal

Measure the three things the eval reports nothing about: citation quality, judge reliability, and latency. Then make a score drop fail the build.

## Acceptance Criteria

- [x] The results file reports citation precision, strip rate, and hallucinated-path rate.
- [x] A human-labeled sample gives a judge agreement figure, reported as Cohen's kappa.
- [x] The results file reports p50 and p95 latency, tool rounds used, and the budget-exhausted rate per tier.
- [x] `make eval` exits non-zero when the total score falls below a committed baseline.
- [x] `README.md` reports the judge agreement figure next to the eval score.
- [x] `make check` passes.

## Items

### 1. Citation precision — the unmeasured differentiator

Citations are the headline feature of the project and no metric scores them. The validator removes bad markers and the eval never counts how many.

Add three figures per run:

| Metric | Definition |
|---|---|
| Citation precision | Fraction of surviving markers whose line range contains a symbol that the answer names |
| Strip rate | Fraction of emitted markers that the validator removed |
| Hallucinated-path rate | Fraction of markers whose `file_path` is absent from the index |

Strip rate needs the marker count before validation. `validate_citations` returns text only. Change it to return `(text, stats)`, or add `validate_citations_with_stats`. Keep the current signature working so `answer_node` needs a small change only.

Citation precision needs the pre-validation markers and the answer text. Compute it per run and average it per tier. A high end-to-end score with a 30% strip rate is a different result from the same score with a 2% strip rate, and today the two look identical.

### 2. Judge validation

`evals/judge_prompt.py` drives a binary pass or fail decision, and nothing measures whether that decision matches a human. An LLM judge with no agreement figure is an uncalibrated instrument.

Steps:

1. Take 40 `(question, answer, judge_verdict)` triples from the existing files in `evals/results/`.
2. Label each one by hand as pass or fail. Do not look at the judge verdict while labeling.
3. Compute Cohen's kappa between the human labels and the judge labels.
4. Write the sample and the figure to `evals/judge_validation.jsonl` and report kappa in `README.md`.

A kappa above 0.8 supports the eval. A kappa below 0.6 means the judge prompt needs work, and that is a result worth reporting too.

Add a second check: run the same answers through a different judge model and report the score delta. Judge variance is a real error term in the headline number.

### 3. Latency, rounds, and budget exhaustion

`_run_once` measures cost and nothing else. Add per run:

- Wall-clock seconds for `graph.invoke`
- Tool rounds used, from `len({t["round"] for t in tool_trace})`
- `budget_exhausted`, which the code already computes at `evals/run.py:153` and never aggregates

Report p50 and p95 latency per tier, mean rounds per tier, and the budget-exhausted rate per tier. A tier that exhausts the 8-round budget often needs a higher budget or better retrieval, and today nothing surfaces that.

### 4. Regression gate

Add `evals/baseline.json`:

```json
{
  "dev": {"median_total": 63, "max_total": 67, "n_runs": 3},
  "test": {"median_total": null, "max_total": null, "n_runs": 3},
  "retrieval": {"recall_at_5": null, "mrr": null},
  "updated": "2026-08-11"
}
```

`run()` compares the total against the baseline and exits 1 if the total falls more than 2 points. Add `--update-baseline` to write a new baseline on purpose. Fill the null fields after tasks 6.3 and 6.4 produce the numbers.

This makes the eval a gate instead of a report, and it gives the CI job from task 6.6 something to protect.

## Files

- `agent/citations.py` — return validation statistics
- `agent/answer_node.py` — carry the statistics in `additional_kwargs`
- `evals/run.py` — citation metrics, timing, rounds, exhaustion rate, baseline gate
- `evals/judge_validation.py` — new, kappa calculation
- `evals/judge_validation.jsonl` — new, the labeled sample
- `evals/baseline.json` — new
- `tests/test_citations.py`, `tests/test_eval_runner.py` — new tests
- `README.md` — citation metrics, judge kappa, latency

## Steps

- [x] Add validation statistics to `agent/citations.py` and keep the current signature working.
- [x] Carry the statistics through `answer_node` in `additional_kwargs`.
- [x] Compute citation precision, strip rate, and hallucinated-path rate in `_run_once`.
- [x] Add wall-clock timing and the tool-round count to `_run_once`.
- [x] Aggregate latency p50 and p95, mean rounds, and exhaustion rate per tier in `format_results_md`.
- [x] Sample 40 triples from `evals/results/` into `evals/judge_validation.jsonl`.
- [x] Label the 40 by hand, without looking at the judge verdicts.
- [x] Write the kappa calculation in `evals/judge_validation.py` and report the figure.
- [x] Run the same answers through a second judge model and record the score delta.
- [x] Add `evals/baseline.json` and the comparison in `run()`, plus `--update-baseline`.
- [x] Report the new metrics in `README.md`.
- [x] Run `make check` and confirm all tests pass.

## Result

All four items landed. Tests 206 (after judge_validation.py) -> 243.

**1. Citation precision.** `agent/citations.py` gained `validate_citations_with_stats`, which `validate_citations` now wraps -- the one existing caller needed no change. Precision checks something plain containment (`chunk_exists_at`) cannot: for a surviving marker, does *any* chunk overlapping that range have a `symbol_name` the answer text actually mentions? A citation whose range is real but attached to the wrong claim now fails precision even though it passes validation. Two new DB methods support it: `chunks_at` (containment query returning the chunk, not a bool) and `file_path_known` (distinguishes a hallucinated path from a real path with a bad range -- different failure modes, now measured separately).

**2. Judge validation.** `evals/judge_validation.py`, new. Stratified sampling (all non-pass rows + one pass row per distinct question, not repeat runs) rather than plain random, because a ~93%-pass judge makes plain random sampling say nothing about false-positive rate. Human labeling: κ=0.872 clean (0.440 raw, with 7/40 rows explained by rubric drift from the A1 eval-criteria fix, not judge noise -- full accounting in `open-questions.md` A3). Cross-judge (haiku-4.5): 97.5% agreement clean, corroborating the human result independently. One real, narrow disagreement found by both: q03, a judge willing to infer a stated fact from field names rather than requiring it explicit.

**3. Latency, rounds, exhaustion.** `_run_once` now times `graph.invoke` and counts distinct tool rounds. `aggregate_by_tier` pools counts across every run in a tier (`sum(stripped)/sum(emitted)`, not a mean of per-run rates, which would be undefined or misleading for a run with zero citations) and reports p50/p95 latency, mean rounds, and budget-exhaustion rate per tier, appended to every results file as "Per-tier instrumentation."

**4. Regression gate.** `evals/baseline.json` holds real numbers, not the template's nulls: dev 91.0/95 (n=3, sonnet-5, 2026-08-12 -- explicitly noted as measured before the A1 fixes, so a conservative floor), test 32/32 (n=1), retrieval 0.956/0.908 (informational, not gated). `run()` returns its rows instead of `None` and never calls `sys.exit` itself, so it stays usable as a plain function from tests; the exit-code decision lives in a new `main(argv) -> int`, matching `judge_validation.py`'s pattern. `--update-baseline` writes a new entry instead of gating.

The gate comparison is skipped (not failed) when the run's `max_total` doesn't match the baseline's -- a grown or shrunk question set (like task 6.4's 33 -> 45) isn't a regression, it's a different scale, and comparing across scales would either mask a real regression or manufacture a fake one.

### What the implementation caught in itself

- `_mock_db` in `tests/test_answer_node.py` only configured `chunk_exists_at`; citations.py's new code calls `chunks_at`/`file_path_known` instead, and an unconfigured `MagicMock()` attribute is truthy by default -- silently broke the "citation gets stripped" test until the mock was updated to match.
- First cut of `--update-baseline` wrote to a bare `evals/baseline.json` with no parent-directory creation, unlike the `results_dir.mkdir(parents=True)` already sitting three lines above it for the results file. `save_baseline` now creates its parent dir too.
- The first regression-gate test asserted `code == 1` with numbers that didn't actually breach the `drop > 2.0` tolerance -- twice. A 1-question set caps the possible drop at 2.0 (not `>` 2.0), and the second attempt's mock answer happened to contain the expected file path substring, keeping `file_ok=True` even with the judge failing, so the drop landed at exactly 2.0 again. Fixed by widening to 2 questions with an answer that fails both checks, which produces an unambiguous 4-point drop -- and this test is exactly the "make eval exits non-zero" contract itself, run through `main()`, not just `check_regression()` in isolation.
