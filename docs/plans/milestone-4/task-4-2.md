# Task 4.2 — Eval Questions

## Goal

Author `evals/questions.jsonl` with 34 hand-crafted questions covering `langchain-core` source, spread across recall / behavior / hard / definition / usage / cross-file / negative tiers.

## Acceptance Criteria

- [x] `evals/questions.jsonl` committed with 34 question entries (plus `_meta` header lines).
- [x] Each entry has all required fields.
- [x] Covers all 7 tiers (recall, behavior, hard, definition, usage, cross-file, negative).
- [x] `expected_file_paths` values are valid short paths (corpus root stripped, matches index).
- [x] Negative-tier questions use `expected_file_paths: []` and `description_must_include` describes correct refusal behavior.

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
| definition | Structure/interface focused — enumerate fields, types, and purposes |
| usage | How to use the API in practice — method signatures, return types, patterns |
| cross-file | Answer requires tracing symbols across ≥2 files |
| negative | Topic is NOT in langchain-core corpus; correct answer is explicit refusal |

For negative questions, `expected_file_paths: []` and `file_ok` is always 0 (N/A). Max achievable score = 1 (judge only). The judge checks that the model admits the answer is not in corpus rather than hallucinating.

## Steps

- [x] Draft 34 questions covering diverse `langchain-core` symbols and concepts.
- [x] Verify `expected_file_paths` values exist in the index.
- [x] Write `evals/questions.jsonl`.
- [x] Add 4 new-tier questions (q31–q34): definition, usage, cross-file, negative.
