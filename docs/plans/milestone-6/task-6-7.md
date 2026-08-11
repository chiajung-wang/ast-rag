# Task 6.7 — Eval Instrumentation

## Goal

Measure the three things the eval reports nothing about: citation quality, judge reliability, and latency. Then make a score drop fail the build.

## Acceptance Criteria

- [ ] The results file reports citation precision, strip rate, and hallucinated-path rate.
- [ ] A human-labeled sample gives a judge agreement figure, reported as Cohen's kappa.
- [ ] The results file reports p50 and p95 latency, tool rounds used, and the budget-exhausted rate per tier.
- [ ] `make eval` exits non-zero when the total score falls below a committed baseline.
- [ ] `README.md` reports the judge agreement figure next to the eval score.
- [ ] `make check` passes.

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

- [ ] Add validation statistics to `agent/citations.py` and keep the current signature working.
- [ ] Carry the statistics through `answer_node` in `additional_kwargs`.
- [ ] Compute citation precision, strip rate, and hallucinated-path rate in `_run_once`.
- [ ] Add wall-clock timing and the tool-round count to `_run_once`.
- [ ] Aggregate latency p50 and p95, mean rounds, and exhaustion rate per tier in `format_results_md`.
- [ ] Sample 40 triples from `evals/results/` into `evals/judge_validation.jsonl`.
- [ ] Label the 40 by hand, without looking at the judge verdicts.
- [ ] Write the kappa calculation in `evals/judge_validation.py` and report the figure.
- [ ] Run the same answers through a second judge model and record the score delta.
- [ ] Add `evals/baseline.json` and the comparison in `run()`, plus `--update-baseline`.
- [ ] Report the new metrics in `README.md`.
- [ ] Run `make check` and confirm all tests pass.
