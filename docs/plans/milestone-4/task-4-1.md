# Task 4.1 — Streamlit UI

## Goal

Build `app.py`: a Streamlit chat interface that invokes the LangGraph agent and renders citation expanders for each `[path:start-end]` marker in the answer.

## Acceptance Criteria

- [x] `make run` launches Streamlit at `localhost:8501` with a chat input.
- [x] Typing a question returns an agent answer displayed with markers intact in the prose.
- [x] Each `[path:start-end]` marker renders as a `st.expander` below the answer containing raw source lines in `st.code` and a GitHub permalink.
- [x] Conversation history persists across turns within the session.
- [x] No crash when answer has zero citation markers.

## Design

**Session state**: `st.session_state.messages` holds `list[BaseMessage]`. Each turn calls `graph.invoke({"messages": history, "retrieved_chunks": []})` — stateless graph, stateful UI.

**Citation expanders**: parse `[path:start-end]` from answer text with regex. For each match call `retrieval.pipeline.read_file(path, start, end)` directly. Render:
1. `st.expander(f"[{path}:{start}-{end}]")`
2. Inside: `st.code(source_lines, language="python")`
3. Below code: GitHub permalink using `COMMIT_SHA` from `indexer.corpus_config`.

**GitHub permalink format**: `https://github.com/langchain-ai/langchain/blob/{COMMIT_SHA}/libs/core/{path}#L{start}-L{end}`

## Files

- `app.py` — new file at repo root

## Steps

- [x] Write `app.py` with session state init, chat input loop, agent invocation.
- [x] Add citation parser (regex `\[([^:\]\s]+):(\d+)-(\d+)\]`) and expander renderer.
- [x] Verify `make run` launches and golden path works (ask "Where is Runnable defined?"). Use `.venv/Scripts/python -m streamlit run app.py` if `make run` doesn't pick up the venv.
- [x] Verify zero-citation answer renders without error.
