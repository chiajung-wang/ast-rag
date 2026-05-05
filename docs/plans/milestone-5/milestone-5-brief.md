# Milestone 5: Polish + Ship

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repo runs from scratch via `make install && make index && make run`. README has real eval results. Error paths are handled gracefully.

**Architecture:** Add error handling to agent (empty retrieval, API errors). Verify `make install` works on a clean environment. Fill eval results into README.

**Prerequisite:** Milestone 4 complete. At least one eval run completed.

---

## Completion Criteria

1. `make install && make index && make run` works on a clean Python 3.11 env (no pre-existing `.db`).
2. Agent returns a graceful message (not a stack trace) when: retrieval returns 0 chunks, OpenAI API key missing, Anthropic API key missing.
3. README `## Eval results` section contains real scores from `evals/results.md`.
4. All unit tests pass: `pytest tests/ -v`.
5. `git status` shows clean working tree after `make index` + `make eval` (generated files gitignored).

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 5.1 | Error handling | Updates to `agent/answer_node.py`, `retrieval/pipeline.py`, `app.py` |
| 5.2 | README eval results + final integration | `README.md` update, `Makefile` smoke test |
