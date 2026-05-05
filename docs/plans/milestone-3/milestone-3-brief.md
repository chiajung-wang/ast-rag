# Milestone 3: Agent

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a 2-node LangGraph agent (`retrieve → answer`) so that `python ask.py "Where is Runnable defined?"` returns a streamed answer with validated citations.

**Architecture:** `AgentState` TypedDict holds `messages` and `retrieved_chunks`. The `retrieve` node calls `retrieval.pipeline.retrieve`. The `answer` node calls Claude Sonnet 4.6 with a hard citation requirement in the system prompt; it may call `read_file` as a tool (up to 3 rounds). Citation validator parses and strips unverifiable `[path:start-end]` markers before returning.

**Tech Stack:** `langgraph`, `langchain-anthropic` (`claude-sonnet-4-6`), `langchain-core`

**Prerequisite:** Milestone 2 complete.

---

## Completion Criteria

1. All unit tests pass: `pytest tests/ -v`.
2. `python ask.py "Where is RunnableSequence defined?"` returns an answer containing a `[runnables/base.py:N-M]` citation.
3. `python ask.py "What is RunnablePassthrough?"` returns an answer with ≥1 citation.
4. Answers never contain citations that fail `db.chunk_exists_at` (validator removes them).
5. `python ask.py "foo bar baz nonsense"` returns a graceful "I don't have source for this" response, not an error.

---

## Tasks

| # | Task | What it builds |
|---|---|---|
| 3.1 | Agent state + retrieve node | `agent/state.py`, `agent/retrieve_node.py`, `tests/test_retrieve_node.py` |
| 3.2 | Citation validator | `agent/citations.py`, `tests/test_citations.py` |
| 3.3 | Answer node + graph | `agent/answer_node.py`, `agent/graph.py`, `tests/test_answer_node.py` |
| 3.4 | `ask.py` CLI | `ask.py` |

---

## What Milestone 4 Depends On

- `agent.graph.graph` — compiled LangGraph instance
- `agent.graph.graph.invoke({"messages": [...], "retrieved_chunks": []})` → dict with `messages`
