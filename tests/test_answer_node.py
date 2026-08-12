from unittest.mock import patch, MagicMock
import pytest
import httpx
import anthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from storage.chunk import make_chunk
from agent.state import AgentState
from agent.answer_node import reset_model_cache


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """answer_node caches one client per model for the life of the process.
    Without this, a patched ChatAnthropic from one test is reused by the next."""
    reset_model_cache()
    yield
    reset_model_cache()


def _fake_api_error(msg: str = "connection failed") -> anthropic.APIError:
    req = httpx.Request("GET", "https://api.anthropic.com")
    return anthropic.APIConnectionError(message=msg, request=req)


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


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_api_error_returns_graceful_message(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.side_effect = _fake_api_error("connection refused")
    mock_anthropic.return_value = mock_model

    state = AgentState(
        messages=[HumanMessage("Where is Runnable?")],
        retrieved_chunks=[],
    )

    from agent.answer_node import answer_node
    result = answer_node(state)

    assert len(result["messages"]) == 2
    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "Anthropic API error" in last.content
    assert "try again" in last.content


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_empty_chunks_no_crash(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)
    response = AIMessage(content="I don't have source for this.")
    mock_anthropic.return_value = _mock_model([response])

    state = AgentState(
        messages=[HumanMessage("What is Runnable?")],
        retrieved_chunks=[],
    )

    from agent.answer_node import answer_node
    result = answer_node(state)

    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)


# ── prompt caching wiring (task 6.6) ─────────────────────────────────────────

def test_system_message_is_a_cacheable_block():
    """The tool loop resends this up to 8 times; the chunk text is the bulk."""
    from agent.answer_node import _build_system_message
    msg = _build_system_message([_make_chunk("RunnableSequence")])
    assert isinstance(msg.content, list) and len(msg.content) == 1
    block = msg.content[0]
    assert block["type"] == "text"
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "RunnableSequence" in block["text"]


def test_system_message_cacheable_even_with_no_chunks():
    from agent.answer_node import _build_system_message
    block = _build_system_message([]).content[0]
    assert block["cache_control"] == {"type": "ephemeral"}
    assert "(no chunks retrieved)" in block["text"]


@patch("agent.answer_node.ChatAnthropic")
def test_model_client_is_reused_across_calls(mock_anthropic):
    """Rebuilding the client per round threw away the connection pool."""
    from agent.answer_node import _get_model
    mock_anthropic.return_value = _mock_model([])
    a = _get_model("claude-haiku-4-5")
    b = _get_model("claude-haiku-4-5")
    assert a is b
    assert mock_anthropic.call_count == 1


@patch("agent.answer_node.ChatAnthropic")
def test_model_cache_separates_tool_bound_from_plain(mock_anthropic):
    """The budget-exhausted path needs a client with no tools bound.

    Asserted on the cache keys, not object identity: the mock's bind_tools
    returns the mock itself, so identity cannot tell the two apart.
    """
    from agent.answer_node import _get_model, _MODELS
    mock_anthropic.return_value = _mock_model([])
    _get_model("claude-haiku-4-5", with_tools=True)
    _get_model("claude-haiku-4-5", with_tools=False)
    assert set(_MODELS) == {"claude-haiku-4-5:tools", "claude-haiku-4-5:plain"}


@patch("agent.answer_node.ChatAnthropic")
def test_model_cache_separates_models(mock_anthropic):
    from agent.answer_node import _get_model
    mock_anthropic.return_value = _mock_model([])
    _get_model("claude-haiku-4-5")
    _get_model("claude-sonnet-4-6")
    assert mock_anthropic.call_count == 2


@patch("agent.answer_node.ChatAnthropic")
@patch("agent.answer_node._get_db")
def test_answer_node_reports_cache_tokens(mock_get_db, mock_anthropic):
    mock_get_db.return_value = _mock_db(exists=True)
    response = AIMessage(
        content="Answer [runnables/base.py:10-50]",
        usage_metadata={
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
            "input_token_details": {"cache_read": 4096, "cache_creation": 0},
        },
    )
    mock_anthropic.return_value = _mock_model([response])
    from agent.answer_node import answer_node
    state = AgentState(messages=[HumanMessage("q")], retrieved_chunks=[_make_chunk("X")])
    out = answer_node(state)["messages"][-1]
    assert out.additional_kwargs["cache_read_tokens"] == 4096
