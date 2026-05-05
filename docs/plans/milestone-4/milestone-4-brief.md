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
4. `evals/questions.jsonl` committed with exactly 20 questions (8 easy / 8 medium / 4 hard).
5. `make eval` completes without error and writes `evals/results.md`.
6. `evals/results.md` contains per-question scores and a summary line.

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 4.1 | Streamlit UI | `app.py` |
| 4.2 | Eval questions | `evals/questions.jsonl` |
| 4.3 | Eval runner | `evals/run.py`, `tests/test_eval_runner.py` |

---

## What Milestone 5 Depends On

- `make run` working
- `make eval` working
- Baseline eval score (needed for README)
