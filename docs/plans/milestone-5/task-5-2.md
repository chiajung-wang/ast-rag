# Task 5.2 — README Eval Results + `make check` Smoke Target

## Goal

Re-run the full 34-question eval at n=1, update README with the real score, add a `make check` target, and add a `## Development` section to README.

**Blocked by: Task 5.1** (so `make check` runs against error-safe code).

## Acceptance Criteria

- [ ] `make eval` runs all 34 questions at n=1 and writes a fresh results file to `evals/results/`.
- [ ] `README.md` `## Eval results` single summary line updated with the real score from that run (format: `**X / 67 (Y%)** — haiku-4-5 agent, sonnet-4-6 judge, n=1 across 34 questions`).
- [ ] `Makefile` has a `check` target (`pytest tests/ -v`) and `check` is in `.PHONY`.
- [ ] `README.md` has a `## Development` section (below Setup) documenting both `make check` and `make eval`.
- [ ] `git status` is clean after `make index` + `make eval` (manual verification — `.db` and `evals/results/` already gitignored).
- [ ] `pytest tests/ -v` passes.

## Design

**Re-run eval:**
```bash
python evals/run.py --runs 1
```
Default `make eval` runs n=3. Pass `--n 1` explicitly to match baseline methodology and keep cost ~$0.15.

**`make check` target:**
```makefile
.PHONY: install run index eval check

check:
	pytest tests/ -v
```
Tests only — no git status check (would false-positive on WIP).

**README `## Development` section** (add below Setup, above Eval results):
```markdown
## Development

```bash
make check   # run unit tests
make eval    # run 34-question eval, write results to evals/results/
```
```

**README `## Eval results`** — update single line only, no table:
```markdown
Baseline: **X / 67 (Y%)** — haiku-4-5 agent, sonnet-4-6 judge, n=1 across 34 questions (7 tiers: recall / behavior / hard / definition / usage / cross-file / negative).
```

## Context

Only one results file exists (`results-0511-1703-haiku-4-5-sonnet-4-6.md`), containing 10/34 questions — the eval was truncated. The README's `63/67 (94%)` came from an earlier run no longer on disk. Re-running is required to get a verifiable number.

## Files

- `Makefile` — add `check` target, update `.PHONY`
- `README.md` — add `## Development` section; update `## Eval results` summary line

## Steps

- [ ] Add `check` target to `Makefile` and update `.PHONY`.
- [ ] Run `python evals/run.py --runs 1` — wait for all 34 questions to complete.
- [ ] Update `## Eval results` summary line in `README.md` with real score from the new results file.
- [ ] Add `## Development` section to `README.md` with `make check` and `make eval`.
- [ ] Run `make check` — confirm all tests pass.
- [ ] Run `git status` — confirm clean (no untracked generated files from `make index` or `make eval`).
