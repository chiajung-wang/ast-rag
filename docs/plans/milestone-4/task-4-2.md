# Task 4.2 — Eval Questions

## Goal

Author `evals/questions.jsonl` with exactly 20 hand-crafted questions covering `langchain-core` source, spread across easy / medium / hard difficulty.

## Acceptance Criteria

- [ ] `evals/questions.jsonl` committed with exactly 20 entries.
- [ ] Each entry has all four required fields (no extras needed).
- [ ] Distribution: 8 easy, 8 medium, 4 hard.
- [ ] `expected_file_path` values are valid short paths (corpus root stripped, matches index).

## Schema

Each line is a JSON object:

```json
{
  "id": "q01",
  "question": "Where is RunnableSequence defined?",
  "expected_file_path": "runnables/base.py",
  "ground_truth_answer": "RunnableSequence is defined in runnables/base.py. It is the core composition primitive created when you chain runnables with the | operator."
}
```

- `id` — `q01`–`q20`, zero-padded
- `question` — natural-language query as a user would type it
- `expected_file_path` — short path (corpus root stripped), used for auto file-path check
- `ground_truth_answer` — reference answer for LLM-as-judge scoring

## Difficulty Guidelines

| Difficulty | Count | Characteristics |
|---|---|---|
| easy | 8 | Single well-known symbol, unambiguous file |
| medium | 8 | Concept spans multiple symbols or requires reading docstrings |
| hard | 4 | Subtle behavior, inheritance chain, or requires cross-file reasoning |

## Steps

- [ ] Draft 20 questions covering diverse `langchain-core` symbols and concepts.
- [ ] Verify `expected_file_path` values exist in the index (run `python query.py` spot checks).
- [ ] Write `evals/questions.jsonl`.
