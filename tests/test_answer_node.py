from unittest.mock import patch, MagicMock
import pytest
import httpx
import openai
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from storage.chunk import make_chunk
from agent.state import AgentState
from agent.answer_node import reset_model_cache


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """answer_node caches one client per model for the life of the process.
    Without this, a patched ChatOpenAI from one test is reused by the next."""
    reset_model_cache()
    yield
    reset_model_cache()


def _fake_api_error(msg: str = "connection failed") -> openai.APIError:
    """OpenRouter is reached through the OpenAI client, so provider failures
    arrive as openai.APIError. Catching the wrong exception type here is how
    the graceful-error path silently regresses to a traceback."""
    req = httpx.Request("GET", "https://openrouter.ai/api/v1")
    return openai.APIConnectionError(message=msg, request=req)


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
    # agent/citations.py validates via chunks_at (list) + file_path_known,
    # not the boolean chunk_exists_at, since task 6.7's precision metric
    # needs the actual chunk to check its symbol_name. An unconfigured
    # MagicMock().chunks_at(...) returns a truthy mock either way, which
    # silently broke the "citation gets stripped" test until this was added.
    mock_db.chunks_at.return_value = [MagicMock(symbol_name="X")] if exists else []
    mock_db.file_path_known.return_value = exists
    return mock_db


@patch("agent.answer_node.ChatOpenAI")
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


@patch("agent.answer_node.ChatOpenAI")
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


@patch("agent.answer_node.ChatOpenAI")
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


@patch("agent.answer_node.ChatOpenAI")
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


@patch("agent.answer_node.ChatOpenAI")
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
    assert "OpenRouter API error" in last.content
    assert "try again" in last.content


@patch("agent.answer_node.ChatOpenAI")
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


@patch("agent.answer_node.ChatOpenAI")
def test_model_client_is_reused_across_calls(mock_anthropic):
    """Rebuilding the client per round threw away the connection pool."""
    from agent.answer_node import _get_model
    mock_anthropic.return_value = _mock_model([])
    a = _get_model("anthropic/claude-haiku-4.5")
    b = _get_model("anthropic/claude-haiku-4.5")
    assert a is b
    assert mock_anthropic.call_count == 1


@patch("agent.answer_node.ChatOpenAI")
def test_model_cache_separates_tool_bound_from_plain(mock_anthropic):
    """The budget-exhausted path needs a client with no tools bound.

    Asserted on the cache keys, not object identity: the mock's bind_tools
    returns the mock itself, so identity cannot tell the two apart.
    """
    from agent.answer_node import _get_model, _MODELS
    mock_anthropic.return_value = _mock_model([])
    _get_model("anthropic/claude-haiku-4.5", with_tools=True)
    _get_model("anthropic/claude-haiku-4.5", with_tools=False)
    assert set(_MODELS) == {
        "anthropic/claude-haiku-4.5:tools", "anthropic/claude-haiku-4.5:plain"}


@patch("agent.answer_node.ChatOpenAI")
def test_model_cache_separates_models(mock_anthropic):
    from agent.answer_node import _get_model
    mock_anthropic.return_value = _mock_model([])
    _get_model("anthropic/claude-haiku-4.5")
    _get_model("anthropic/claude-sonnet-4.6")
    assert mock_anthropic.call_count == 2


@patch("agent.answer_node.ChatOpenAI")
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


def test_system_prefix_is_byte_stable_across_tool_rounds():
    """Caching is a prefix match, so the block must not vary between rounds.

    The API decides whether a hit occurs (it declines below 4096 tokens on
    Haiku 4.5). Prefix stability is the part this code controls, so that is
    what gets asserted here.
    """
    from agent.answer_node import answer_node
    seen_systems = []

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "read_file",
                     "args": {"path": "runnables/base.py", "line_start": 1, "line_end": 5},
                     "id": "t1"}],
    )
    final = AIMessage(content="Done [runnables/base.py:10-50]")

    model = MagicMock()
    model.bind_tools.return_value = model

    def record(msgs, *a, **k):
        seen_systems.append(msgs[0].content)
        return tool_call if len(seen_systems) == 1 else final

    model.invoke.side_effect = record

    with patch("agent.answer_node.ChatOpenAI", return_value=model), \
         patch("agent.answer_node._get_db", return_value=_mock_db(True)), \
         patch("agent.answer_node._read_file", return_value="source"):
        state = AgentState(messages=[HumanMessage("q")],
                           retrieved_chunks=[_make_chunk("RunnableSequence")])
        answer_node(state)

    assert len(seen_systems) == 2, "expected a tool round then a final round"
    assert seen_systems[0] == seen_systems[1], "system prefix changed between rounds"
    assert seen_systems[0][0]["cache_control"] == {"type": "ephemeral"}


# ── malformed tool-call args (2026-08-12 finding) ────────────────────────────

@patch("agent.answer_node.ChatOpenAI")
@patch("agent.answer_node._get_db")
def test_malformed_tool_args_fed_back_not_crashed(mock_get_db, mock_anthropic):
    """A model that packs "path:start-end" into read_file's path field (instead
    of separate path/line_start/line_end) used to raise an uncaught pydantic
    ValidationError that killed graph.invoke entirely. It must now come back
    as an error ToolMessage the model can recover from, same as a bad path or
    bad line range already do."""
    mock_get_db.return_value = _mock_db(exists=True)
    bad_call = AIMessage(
        content="",
        tool_calls=[{
            "name": "read_file",
            # real read_file requires path, line_start, line_end -- this omits
            # the last two, which is exactly the shape sonnet-5 produced.
            "args": {"path": "language_models/chat_models.py:1767-1926"},
            "id": "tc1",
            "type": "tool_call",
        }],
    )
    final = AIMessage(content="Answer [runnables/base.py:10-50]")
    mock_anthropic.return_value = _mock_model([bad_call, final])

    state = AgentState(messages=[HumanMessage("q")], retrieved_chunks=[])

    from agent.answer_node import answer_node
    result = answer_node(state)  # must not raise

    # answer_node returns only the final AIMessage, not the intermediate
    # ToolMessage -- the recovery is verified by the loop completing at all
    # (no raise) and reaching the model's post-error final answer.
    assert len(result["messages"]) == 2
    assert "Answer" in result["messages"][-1].content
    assert result["messages"][-1].additional_kwargs["tool_trace"][0]["tool"] == "read_file"


# ── unhandled SDK exceptions in the model-invoke loop (2026-08-12 A1 finding) ─

@patch("agent.answer_node.ChatOpenAI")
@patch("agent.answer_node._get_db")
def test_non_api_exception_from_model_invoke_handled_gracefully(mock_get_db, mock_anthropic):
    """A1's n=3 dev run hit json.JSONDecodeError twice -- langchain_openai
    parsing a malformed tool-call-arguments string somewhere inside
    model.invoke(). Not an openai.APIError, so the old except openai.APIError
    let it escape entirely: score=0, agent_cost=$0.00, no message, no tool
    trace -- indistinguishable from a hang. Any exception in this scope is
    provider/SDK territory, not our own logic, so the catch must be broad."""
    import json as json_module
    mock_get_db.return_value = _mock_db(exists=True)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.side_effect = json_module.JSONDecodeError("Expecting value", "doc", 0)
    mock_anthropic.return_value = mock_model

    state = AgentState(messages=[HumanMessage("q")], retrieved_chunks=[])

    from agent.answer_node import answer_node
    result = answer_node(state)  # must not raise

    assert len(result["messages"]) == 2
    last = result["messages"][-1]
    assert "Unexpected error" in last.content
    assert "JSONDecodeError" in last.content


@patch("agent.answer_node.ChatOpenAI")
@patch("agent.answer_node._get_db")
def test_openai_api_error_still_gets_its_specific_message(mock_get_db, mock_anthropic):
    """The broader catch must not swallow the more specific, more actionable
    openai.APIError message (credit balance, rate limits) behind the generic
    fallback -- except ordering, not just presence, is what's being tested."""
    mock_get_db.return_value = _mock_db(exists=True)
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.invoke.side_effect = _fake_api_error("credit balance too low")
    mock_anthropic.return_value = mock_model

    state = AgentState(messages=[HumanMessage("q")], retrieved_chunks=[])

    from agent.answer_node import answer_node
    result = answer_node(state)

    last = result["messages"][-1]
    assert "OpenRouter API error" in last.content
    assert "Unexpected error" not in last.content
