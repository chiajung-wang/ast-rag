# Task 3.1: Agent State + Retrieve Node

**What to build:** `AgentState` TypedDict defining the graph's shared state, and `retrieve_node` that pulls the last user message and calls `retrieval.pipeline.retrieve` to populate `retrieved_chunks`.

**Blocked by:** Milestone 2 complete (retrieval pipeline exists)

**Acceptance criteria:**
- [ ] `AgentState` has `messages: list[BaseMessage]` and `retrieved_chunks: list[Chunk]`
- [ ] `retrieve_node` extracts last message content, calls `retrieve(query)`, returns `{"retrieved_chunks": chunks}`
- [ ] `retrieve_node` uses the last message (not first) as the query
- [ ] Empty retrieve result → `retrieved_chunks = []` (no error)
- [ ] All tests pass

---

**Files:**
- Create: `agent/state.py`
- Create: `agent/retrieve_node.py`
- Create: `tests/test_retrieve_node.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieve_node.py`:

```python
from unittest.mock import patch
from langchain_core.messages import HumanMessage
from agent.state import AgentState
from agent.retrieve_node import retrieve_node
from storage.chunk import make_chunk


def _make_chunk(name: str):
    return make_chunk("f.py", name, "function", None, 1, 5, None, f"def {name}(): pass")


def test_retrieve_node_calls_retrieve_with_last_message():
    chunk = _make_chunk("RunnableSequence")
    with patch("agent.retrieve_node.retrieve", return_value=[chunk]) as mock:
        state = AgentState(
            messages=[HumanMessage("Where is RunnableSequence?")],
            retrieved_chunks=[],
        )
        retrieve_node(state)
    mock.assert_called_once_with("Where is RunnableSequence?")


def test_retrieve_node_returns_chunks():
    chunk = _make_chunk("RunnableSequence")
    with patch("agent.retrieve_node.retrieve", return_value=[chunk]):
        state = AgentState(
            messages=[HumanMessage("Where is RunnableSequence?")],
            retrieved_chunks=[],
        )
        result = retrieve_node(state)
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0].symbol_name == "RunnableSequence"


def test_retrieve_node_uses_last_message():
    with patch("agent.retrieve_node.retrieve", return_value=[]) as mock:
        state = AgentState(
            messages=[HumanMessage("first"), HumanMessage("second query")],
            retrieved_chunks=[],
        )
        retrieve_node(state)
    mock.assert_called_once_with("second query")


def test_retrieve_node_empty_results():
    with patch("agent.retrieve_node.retrieve", return_value=[]):
        state = AgentState(
            messages=[HumanMessage("nonsense query")],
            retrieved_chunks=[],
        )
        result = retrieve_node(state)
    assert result["retrieved_chunks"] == []
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_retrieve_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.state'`

- [ ] **Step 3: Write `agent/state.py` and `agent/retrieve_node.py`**

`agent/state.py`:

```python
from __future__ import annotations
from typing import TypedDict
from langchain_core.messages import BaseMessage
from storage.chunk import Chunk


class AgentState(TypedDict):
    messages: list[BaseMessage]
    retrieved_chunks: list[Chunk]
```

`agent/retrieve_node.py`:

```python
from __future__ import annotations
from agent.state import AgentState
from retrieval.pipeline import retrieve


def retrieve_node(state: AgentState) -> dict:
    query = state["messages"][-1].content
    chunks = retrieve(query)
    return {"retrieved_chunks": chunks}
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_retrieve_node.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/state.py agent/retrieve_node.py tests/test_retrieve_node.py
git commit -m "feat(agent): AgentState and retrieve node"
```
