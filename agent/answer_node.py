from __future__ import annotations
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from storage.db import DB
from retrieval.pipeline import read_file as _read_file
from agent.state import AgentState
from agent.citations import validate_citations
from indexer.corpus_config import DB_PATH

MAX_TOOL_ROUNDS = 5

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
        "STEP 1 — Before writing your answer, call read_file on the relevant source file "
        "to read the complete class or function definition. Retrieved chunks may be truncated "
        "and miss critical details. Read enough lines to see the full class body.\n\n"
        "STEP 1b — If the class or function references other types (e.g. TypedDicts, "
        "dataclasses, field types, parent classes defined elsewhere), call read_file on "
        "those files too to discover their fields and structure.\n\n"
        "STEP 1c — To understand sync vs async execution differences, read the .invoke() "
        "and .ainvoke() (or equivalent) method bodies, not just the class-level docstring.\n\n"
        "STEP 2 — When answering, be exhaustive. Enumerate:\n"
        "- All fields / attributes and their types (read referenced TypedDicts/dataclasses for sub-fields)\n"
        "- All abstract or required methods subclasses must implement\n"
        "- Both sync and async method variants (e.g. invoke/ainvoke, on_*/aon_*)\n"
        "- Configuration flags and their effect on runtime behavior (check parent classes too)\n"
        "- Sync vs async execution differences (e.g. thread pool for sync, coroutine for async)\n\n"
        "For every symbol or concept: (1) cite with [path:start-end], "
        "(2) explain purpose, key interface, and relation to other components. "
        "Never state a fact without a citation. "
        'If source does not support a claim, say "I don\'t have source for this".\n\n'
        f"Retrieved source chunks:\n{chunk_context}"
    )


def answer_node(state: AgentState) -> dict:
    model_name = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")
    model = ChatAnthropic(model=model_name).bind_tools([read_file])
    system = SystemMessage(content=_build_system_prompt(state["retrieved_chunks"]))
    messages: list = [system] + list(state["messages"])

    response = None
    total_input_tokens = 0
    total_output_tokens = 0

    def _add_usage(r):
        nonlocal total_input_tokens, total_output_tokens
        u = r.usage_metadata or {}
        total_input_tokens += u.get("input_tokens", 0)
        total_output_tokens += u.get("output_tokens", 0)

    for _ in range(MAX_TOOL_ROUNDS):
        response = model.invoke(messages)
        _add_usage(response)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = read_file.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    else:
        messages.append(HumanMessage(content=(
            "Tool budget exhausted. Write your final answer NOW using only the "
            "context already gathered. Do not request any more tools. "
            "Cite with [path:start-end] for every claim."
        )))
        model_no_tools = ChatAnthropic(model=model_name)
        response = model_no_tools.invoke(messages)
        _add_usage(response)

    content = response.content
    if not isinstance(content, str):
        content = "".join(
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for b in content
        )
    validated = validate_citations(content, _get_db())
    final = AIMessage(
        content=validated,
        usage_metadata={"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
    )
    return {"messages": list(state["messages"]) + [final]}
