# Milestone 4: UI + Eval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Streamlit chat UI with citation expanders and a working `make eval` that produces `evals/results.md` with per-question 0/1/2 scores.

**Architecture:** Streamlit session state holds message history. Each agent answer is parsed for `[path:start-end]` markers; each marker renders as an `st.expander` with source lines and a GitHub permalink (pinned commit SHA). Eval runner loads `evals/questions.jsonl`, runs each through the agent, scores with a hybrid heuristic (file-path check) + LLM-as-judge (Claude Sonnet 4.6), writes `evals/results.md`.

**Tech Stack:** `streamlit`, `langchain-anthropic` (LLM judge), `jsonlines`

**Prerequisite:** Milestone 3 complete.

---

## Completion Criteria

1. `make run` launches Streamlit. Browser at `localhost:8501` shows a chat input.
2. Typing "Where is Runnable defined?" returns an answer with ≥1 citation expander.
3. Clicking a citation expander shows source code and a GitHub link.
4. `evals/questions.jsonl` committed with 34 questions across 6 tiers (recall / behavior / hard / definition / usage / cross-file / negative).
5. `make eval` completes without error and writes `evals/results/results-<timestamp>-<agent>-<judge>.md`.
6. Results file contains per-question median score, variance, file_ok%, judge%, separate agent/judge costs, and per-run tool traces.

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 4.1 | Streamlit UI | `app.py` |
| 4.2 | Eval questions | `evals/questions.jsonl` |
| 4.3 | Eval runner | `evals/run.py`, `tests/test_eval_runner.py` |

---

## Baseline Eval Results (haiku-4-5 agent, sonnet-4-6 judge, n=1)

63 / 67 (94%) across 34 questions. Failures: q01 (recall, misses `|` operator mention), q16/q20 (behavior, judge variance), q29 (hard, misses community-vs-core boundary claim).

## What Milestone 5 Depends On

- `make run` working
- `make eval` working
- Baseline eval score (needed for README)
