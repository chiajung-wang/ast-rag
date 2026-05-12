# Task 5.1 — Error Handling

## Goal

Add graceful error handling so no user-visible stack trace can escape from: missing API keys, missing DB, empty retrieval, or Anthropic API failures. All surfaces return a friendly message.

## Acceptance Criteria

- [ ] `app.py` checks `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `DB_PATH` existence at startup; shows `st.error(...)` + `st.stop()` for each missing one.
- [ ] `answer_node.py` catches `anthropic.APIError` (base class — covers auth, rate limit, connection errors) and returns an `AIMessage` with a generic actionable message instead of a traceback.
- [ ] Empty retrieval (`retrieved_chunks == []`) does not crash — verified by test (no code change; `_build_system_prompt([])` already handles it).
- [ ] Two unit tests added to `tests/test_answer_node.py`: (1) `anthropic.APIError` → graceful AIMessage; (2) `retrieved_chunks=[]` → no crash, answer returned.
- [ ] No tests added for `app.py` key/DB guard (trivial `os.getenv` + `os.path.exists` — no logic to test).
- [ ] `pytest tests/ -v` passes.

## Design

**`app.py` startup guard** — after `load_dotenv()`, before any UI:
```python
import os
from indexer.corpus_config import DB_PATH

for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    if not os.getenv(var):
        st.error(f"Missing env var: {var}. Add it to .env and restart.")
        st.stop()

if not os.path.exists(DB_PATH):
    st.error("Index not found. Run `make index` first.")
    st.stop()
```

**`answer_node.py` error catch** — single `try/except` wrapping the entire loop + budget-exhausted else branch:
```python
import anthropic

try:
    for round_num in range(MAX_TOOL_ROUNDS):
        ...
    else:
        ...  # budget exhausted path
except anthropic.APIError as e:
    return {"messages": list(state["messages"]) + [
        AIMessage(content=(
            f"Anthropic API error — try again. "
            f"If it persists, check your API key and rate limits. ({e})"
        ))
    ]}
```

**Empty retrieval** — `_build_system_prompt([])` already emits `(no chunks retrieved)`. LLM responds gracefully. No early return — LLM can still answer from training context. Test verifies no crash.

## Scope

- `app.py` only (not `ask.py` CLI — dev tool, out of scope).
- No test for `app.py` guard — logic-free.
- No OpenAI error handling at runtime — startup key check covers the common failure mode.

## Files

- `app.py` — add startup guards after `load_dotenv()`
- `agent/answer_node.py` — wrap tool loop in `try/except anthropic.APIError`
- `tests/test_answer_node.py` — add two tests

## Steps

- [ ] Add startup guards to `app.py` (key check + DB existence check).
- [ ] Add `import anthropic` and outer `try/except anthropic.APIError` to `answer_node.py`.
- [ ] Add `test_answer_node_api_error_returns_graceful_message` to `tests/test_answer_node.py`.
- [ ] Add `test_answer_node_empty_chunks_no_crash` to `tests/test_answer_node.py`.
- [ ] Run `pytest tests/ -v` and confirm all pass.
