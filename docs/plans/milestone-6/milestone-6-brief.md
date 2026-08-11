# Milestone 6: Maintenance — Correctness, Honest Claims, Measurable Retrieval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the system correct and keep every public claim true. Milestones 1 to 5 built the system. This milestone maintains it: fix the defects found in the post-ship audit, remove claims the code does not support, and add the measurements that show why the architecture works.

**Why now:** The audit found 4 defects that change results, 4 statements in `README.md` and `CONTEXT.md` that the code does not support, and 1 prompt that is tuned to the eval set. The eval scores answers end to end only. It cannot tell a retrieval failure from a generation failure. Every item below comes from that audit.

**Architecture:** No new subsystems. The scope boundary from `CLAUDE.md` still holds. Work happens inside the existing modules: `retrieval/`, `storage/`, `agent/`, `evals/`.

**Prerequisite:** Milestone 5 complete. `make check` passes (89 tests).

---

## Completion Criteria

1. An exception inside `graph.invoke` no longer stops an eval run.
2. Retrieval returns the same 5 chunks for the same query across separate processes.
3. `find_symbol` picks a chunk by a documented rule, not by SQLite row order.
4. Every price in `evals/run.py` matches the published rate for that model.
5. `README.md` and `CONTEXT.md` describe the behavior that the code has.
6. `make eval` reports retrieval metrics (recall@5, MRR, nDCG@5) next to the end-to-end score.
7. An ablation table compares BM25 only, dense only, RRF, and RRF plus symbol pre-check.
8. A held-out question set exists. The system prompt was frozen before anybody wrote those questions.
9. `get_class_outline` returns inherited methods. The system prompt no longer names langchain-core classes.
10. GitHub Actions runs `make check` on every push.
11. All unit tests pass: `make check`.

---

## Tasks

| # | Task | What it changes | Effort |
|---|---|---|---|
| 6.1 | Correctness fixes | `evals/run.py`, `retrieval/pipeline.py`, `storage/db.py` | ~1h |
| 6.2 | Honest claims | `README.md`, `CONTEXT.md`, `agent/citations.py` | ~1h |
| 6.3 | Retrieval metrics + ablation | `evals/retrieval_eval.py` (new), `evals/questions.jsonl` | ~4h |
| 6.4 | Question set: held-out, tiers, adversarial | `evals/questions-test.jsonl` (new), `evals/questions.jsonl` | ~3h |
| 6.5 | Inheritance-aware class outline | `indexer/chunker.py`, `storage/db.py`, `agent/answer_node.py` | ~4h |
| 6.6 | Repo hygiene | `.github/workflows/`, `.gitignore`, `storage/db.py`, `agent/answer_node.py` | ~2h |
| 6.7 | Eval instrumentation | `evals/run.py`, `evals/judge_validation.py` (new) | ~4h |

## Out of scope — see `open-questions.md`

`open-questions.md` in this folder holds the decisions that this milestone does not make. It covers 3 items that need the owner's budget or time, 3 that are blocked on data from tasks 6.3 and 6.4, 6 measured defects that milestone 6 leaves alone on purpose, and 4 project-level questions.

Discuss that file after tasks 6.1 to 6.7 land. Do not act on it during the milestone. Several items depend on numbers that do not exist yet.

## Order

Run 6.1 and 6.2 first. They are small and they stop the system from reporting wrong numbers.

Run 6.3 and 6.4 next. They give the project its strongest result: measured retrieval instead of an asserted design.

Run 6.5 after 6.4. Task 6.5 removes a prompt hack, so a held-out set must exist first to show that the removal costs nothing.

Run 6.6 and 6.7 last.
