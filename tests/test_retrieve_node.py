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
