# Task 6.1 — Correctness Fixes

## Goal

Fix 5 defects that change results or stop a run. Each one is a small, local change.

## Acceptance Criteria

- [x] An exception inside `graph.invoke` produces one printed error and one zero-score run record. The eval loop continues.
- [x] `retrieve()` returns the same chunk list for the same query across two separate Python processes.
- [x] `find_symbol` applies a documented tie-break rule when several chunks share a name.
- [x] `read_file` clamps `line_start` to 1 or greater.
- [x] The price table in `evals/run.py` matches the published rate for each model.
- [x] `make check` passes.

## Defects

### 1. `UnboundLocalError` kills the eval run — `evals/run.py:132-180`

`tool_trace` is assigned at line 152, inside the `try` block. If `graph.invoke` raises, the `except` block prints the error, and then the `return` statement reads an unbound name. The `UnboundLocalError` escapes `_run_once` and stops the whole run. The error path that exists to keep the run alive is the path that kills it.

Fix: initialize `tool_trace: list[dict] = []` beside the other defaults at line 133.

### 2. Nondeterministic symbol pre-check — `retrieval/pipeline.py:82-95`

```python
candidates = {m.group(1) for m in _SYMBOL_RE.finditer(query)}
for candidate in candidates:
    ...
    break
```

`candidates` is a set of strings. Python randomizes string hashing per process, so set order changes between runs. A query with 2 or more symbol tokens picks a different symbol for the guaranteed `find_symbol` slot on each run. The eval sets `temperature=0` and then measures a retriever that is not deterministic.

Fix: keep query order and remove duplicates.

```python
seen_candidates: set[str] = set()
candidates: list[str] = []
for m in _SYMBOL_RE.finditer(query):
    token = m.group(1)
    if token not in seen_candidates:
        seen_candidates.add(token)
        candidates.append(token)
```

### 3. `find_symbol` returns an arbitrary row — `storage/db.py:94-103`

`WHERE lower(symbol_name) = lower(?) LIMIT 1` has no `ORDER BY`. The current index holds 2414 chunks under 1296 distinct names. Measured collisions: `__init__` 108 times, `get_lc_namespace` 31, `ainvoke` 24, `invoke` 23. Class names collide too (`ToolCall`, `RunInfo`, `Tee`). The pre-check injects that arbitrary chunk ahead of the RRF results.

Fix: add a deterministic tie-break and document it. Prefer a class over a method, prefer a method over a function, then order by `file_path` and `line_start`.

```sql
ORDER BY
    CASE symbol_type WHEN 'class' THEN 0 WHEN 'method' THEN 1 ELSE 2 END,
    file_path, line_start
LIMIT 1
```

This rule is arbitrary but stable. Record the rule in `CONTEXT.md` under **Symbol Lookup**.

### 4. `read_file` has no lower bound — `retrieval/pipeline.py:65-77`

`line_start=0` produces `lines[-1:actual_end]`. The negative start index resolves to the last line, so the slice returns an empty string for a small range, or a window from the end of the file for a large one. Neither raises. Clamp `line_start` to 1 or greater before the slice.

### 5. Wrong prices — `evals/run.py:17-21`

| Model | In code | Published rate (input / output per MTok) |
|---|---|---|
| `claude-haiku-4-5` | 0.80 / 4.00 | **1.00 / 5.00** |
| `claude-sonnet-4-6` | 3.00 / 15.00 | 3.00 / 15.00 (correct) |
| `claude-opus-4-7` | 15.00 / 75.00 | **5.00 / 25.00** |

Every cost figure in `evals/results/` is wrong. Fix the table. Add a comment that records the date of the rates.

## Files

- `evals/run.py` — initialize `tool_trace`, correct `_PRICES`
- `retrieval/pipeline.py` — ordered candidates, clamp `line_start`
- `storage/db.py` — `ORDER BY` in `symbol_lookup`
- `CONTEXT.md` — record the tie-break rule
- `tests/test_pipeline.py`, `tests/test_db.py`, `tests/test_eval_runner.py` — new tests

## Steps

- [x] Initialize `tool_trace = []` at the top of `_run_once` in `evals/run.py`.
- [x] Add a test: `_run_once` with a graph that raises returns a zero-score record and does not raise.
- [x] Replace the `candidates` set with an ordered, deduplicated list in `retrieval/pipeline.py`.
- [x] Add a test: a query with 2 symbol tokens picks the first token in query order.
- [x] Add the `ORDER BY` clause to `symbol_lookup` in `storage/db.py`.
- [x] Add a test: 3 chunks share a name, and `symbol_lookup` returns the class chunk.
- [x] Clamp `line_start` in `read_file` and add a test for `line_start=0`.
- [x] Correct the `_PRICES` table in `evals/run.py` and add the rate date in a comment.
- [x] Record the `find_symbol` tie-break rule in `CONTEXT.md`.
- [x] Run `make check` and confirm all tests pass.
