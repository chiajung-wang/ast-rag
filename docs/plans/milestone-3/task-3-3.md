# Task 3.3: Answer Node + Compiled Graph

**What to build:** `answer_node` that injects retrieved chunks into a system prompt, runs a tool-call loop (max 3 rounds) with `read_file` as a `@tool`, validates citations, and returns the final `AIMessage`. `graph.py` compiles the 2-node `retrieve → answer` LangGraph.

**Blocked by:** Task 3.1 (AgentState + retrieve node), Task 3.2 (citation validator)

**Acceptance criteria:**
- [x] `answer_node` appends an `AIMessage` to `state["messages"]`
- [x] When LLM returns a tool call, `read_file` executes and result feeds back
- [x] Loop stops after ≤3 rounds even if LLM keeps calling tools
- [x] Citation validator runs on final response text
- [x] `graph.invoke({"messages": [...], "retrieved_chunks": []})` returns dict with `messages`
- [x] All tests pass (LLM mocked — no real API calls)

---

**Files:**
- Create: `agent/answer_node.py`
- Create: `agent/graph.py`
- Create: `tests/test_answer_node.py`

---

- [x] **Step 1: Write the failing tests**

`tests/test_answer_node.py`:

```python
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from storage.chunk import make_chunk
from agent.state import AgentState


def _make_chunk(name: str):
    return make_chunk("runnables/base.py", name, "class", None, 10, 50, None, f"class {name}: pass")


def _mock_model(responses: list):
    mock_model = MagicMock()
    mock_model.invoke.side_effect = responses
    mock_model.bind_tools.return_value = mock_model
    return mock_model


def _mock_db(exists: bool = True):
    mock_db = MagicMock()
    mock_db.chunk_exists_at.return_value = exists
    return mock_db


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_no_tool_calls(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)
    response = AIMessage(content="Answer [runnables/base.py:10-50]")
    mock_anthropic.return_value = _mock_model([response])

    state = AgentState(
        messages=[HumanMessage("Where is RunnableSequence?")],
        retrieved_chunks=[_make_chunk("RunnableSequence")],
    )

    from agent.answer_node import answer_node
    result = answer_node(state)

    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert "Answer" in result["messages"][-1].content


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_tool_call_executes_read_file(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)

    tool_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "read_file",
            "args": {"path": "runnables/base.py", "line_start": 10, "line_end": 20},
            "id": "tc1",
            "type": "tool_call",
        }],
    )
    final_response = AIMessage(content="The code is [runnables/base.py:10-50]")
    mock_anthropic.return_value = _mock_model([tool_response, final_response])

    state = AgentState(
        messages=[HumanMessage("Show me the code")],
        retrieved_chunks=[],
    )

    with patch("agent.answer_node.read_file") as mock_rf:
        mock_rf.invoke.return_value = "class RunnableSequence: pass"
        from agent.answer_node import answer_node
        result = answer_node(state)

    assert mock_rf.invoke.call_count == 1
    assert len(result["messages"]) == 2


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_invalid_citation_stripped(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=False)
    response = AIMessage(content="See [fake/path.py:1-5] for details.")
    mock_anthropic.return_value = _mock_model([response])

    state = AgentState(
        messages=[HumanMessage("question")],
        retrieved_chunks=[],
    )

    from agent.answer_node import answer_node
    result = answer_node(state)
    content = result["messages"][-1].content
    assert "[fake/path.py:1-5]" not in content
    assert "could not be verified" in content


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_graph_invoke_returns_messages(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)
    response = AIMessage(content="RunnableSequence is defined in [runnables/base.py:10-50]")
    mock_anthropic.return_value = _mock_model([response])

    with patch("agent.retrieve_node.retrieve", return_value=[_make_chunk("RunnableSequence")]):
        from agent.graph import graph
        result = graph.invoke({
            "messages": [HumanMessage("Where is RunnableSequence?")],
            "retrieved_chunks": [],
        })

    assert "messages" in result
    assert len(result["messages"]) >= 2
    assert isinstance(result["messages"][-1], AIMessage)
```

- [x] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_answer_node.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.answer_node'`

- [x] **Step 3: Write `agent/answer_node.py`**

```python
from __future__ import annotations
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from storage.db import DB
from retrieval.pipeline import read_file as _read_file
from agent.state import AgentState
from agent.citations import validate_citations
from indexer.corpus_config import DB_PATH

MAX_TOOL_ROUNDS = 3

_db: DB | None = None


def _get_db() -> DB:
    global _db
    if _db is None:
        _db = DB(DB_PATH)
    return _db


@tool
def read_file(path: str, line_start: int, line_end: int) -> str:
    """Read source lines from the langchain-core corpus."""
    return _read_file(path, line_start, line_end)


def _build_system_prompt(chunks) -> str:
    if chunks:
        chunk_context = "\n\n".join(
            f"[{c.file_path}:{c.line_start}-{c.line_end}]\n{c.text}"
            for c in chunks
        )
    else:
        chunk_context = "(no chunks retrieved)"
    return (
        "You are a code assistant for the langchain-core codebase.\n\n"
        "You MUST cite every factual claim with [path:start-end]. "
        "Never state a fact without a citation. "
        'If retrieved chunks do not support a claim, say "I don\'t have source for this" '
        "instead of stating it uncited.\n\n"
        f"Retrieved source chunks:\n{chunk_context}"
    )


def answer_node(state: AgentState) -> dict:
    model = ChatAnthropic(model="claude-sonnet-4-6").bind_tools([read_file])
    system = SystemMessage(content=_build_system_prompt(state["retrieved_chunks"]))
    messages: list = [system] + list(state["messages"])

    response = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = model.invoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = read_file.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    else:
        response = model.invoke(messages)

    validated = validate_citations(response.content, _get_db())
    return {"messages": list(state["messages"]) + [AIMessage(content=validated)]}
```

- [x] **Step 4: Write `agent/graph.py`**

```python
from __future__ import annotations
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.retrieve_node import retrieve_node
from agent.answer_node import answer_node

_builder = StateGraph(AgentState)
_builder.add_node("retrieve", retrieve_node)
_builder.add_node("answer", answer_node)
_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "answer")
_builder.add_edge("answer", END)

graph = _builder.compile()
```

- [x] **Step 5: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_answer_node.py -v
```

Expected: all tests PASSED.

- [x] **Step 6: Commit**

```bash
git add agent/answer_node.py agent/graph.py tests/test_answer_node.py
git commit -m "feat(agent): answer node with tool loop and compiled graph"
```
