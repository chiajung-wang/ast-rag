# Task 4.2 — Eval Questions

## Goal

Author `evals/questions.jsonl` with 30 hand-crafted questions covering `langchain-core` source, spread across recall / behavior / hard tiers.

## Acceptance Criteria

- [x] `evals/questions.jsonl` committed with 30 question entries (plus `_meta` header lines).
- [x] Each entry has all required fields.
- [x] Covers recall, behavior, and hard tiers.
- [x] `expected_file_paths` values are valid short paths (corpus root stripped, matches index).

## Schema

Each question line is a JSON object. Lines with `_meta` key are header/instruction lines skipped by the runner.

```json
{
  "id": "q01",
  "question": "Where is RunnableSequence defined?",
  "expected_file_paths": ["runnables/base.py"],
  "description_must_include": ["composition primitive for chaining runnables", "created by the | operator"],
  "description_must_not_assert": [],
  "tier": "recall",
  "subsystem": "runnables"
}
```

- `id` — `q01`–`q30`, zero-padded
- `question` — natural-language query as a user would type it
- `expected_file_paths` — list of acceptable short paths; model is correct if any appears in the answer
- `description_must_include` — list of concepts the answer must cover (paraphrase OK)
- `description_must_not_assert` — list of claims that must NOT appear (automatic judge fail if present)
- `tier` — `recall` | `behavior` | `hard`
- `subsystem` — for coverage tracking

## Tier Guidelines

| Tier | Characteristics |
|---|---|
| recall | Single well-known symbol, unambiguous file |
| behavior | Concept spans multiple symbols or requires reading docstrings |
| hard | Subtle behavior, inheritance chain, disambiguation, or cross-file reasoning |

## Steps

- [x] Draft 30 questions covering diverse `langchain-core` symbols and concepts.
- [x] Verify `expected_file_paths` values exist in the index.
- [x] Write `evals/questions.jsonl`.
