from __future__ import annotations
import os
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
    model_name = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")
    model = ChatAnthropic(model=model_name).bind_tools([read_file])
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

    content = response.content
    if not isinstance(content, str):
        content = "".join(
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for b in content
        )
    validated = validate_citations(content, _get_db())
    return {"messages": list(state["messages"]) + [AIMessage(content=validated)]}
