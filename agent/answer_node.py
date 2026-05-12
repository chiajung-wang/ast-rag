from __future__ import annotations
import os
import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from storage.db import DB
from retrieval.pipeline import read_file as _read_file
from agent.state import AgentState
from agent.citations import validate_citations
from indexer.corpus_config import DB_PATH

MAX_TOOL_ROUNDS = 8

_db: DB | None = None


def _get_db() -> DB:
    global _db
    if _db is None:
        _db = DB(DB_PATH)
    return _db


@tool
def get_class_outline(class_name: str) -> str:
    """Return all method signatures and line ranges for a class.

    Call this before read_file to get a map of which methods exist and where,
    then use read_file on the specific methods you need.
    """
    db = _get_db()
    chunks = db.class_outline(class_name)
    if not chunks:
        return f"No class '{class_name}' found in corpus."
    lines = []
    for c in chunks:
        sig = c.text.splitlines()[0] if c.text else ""
        doc = f"  # {c.docstring.splitlines()[0][:80]}" if c.docstring else ""
        lines.append(f"[{c.file_path}:{c.line_start}-{c.line_end}] {sig}{doc}")
    return "\n".join(lines)


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
        "STEP 1 — Call get_class_outline on the relevant class first. This returns ALL "
        "method signatures and line ranges in one shot — use it to map the class before "
        "reading anything. For standalone functions, call read_file directly.\n\n"
        "STEP 1b — Before reading ANY source lines, batch ALL related get_class_outline calls "
        "in the SAME round as you process the first outline result:\n"
        "  • Async sibling: for Base* classes drop the 'Base' prefix to get the async name "
        "(e.g. BaseCallbackHandler → AsyncCallbackHandler, BaseRunManager → AsyncRunManager). "
        "ALWAYS call get_class_outline on the async sibling — it holds async def versions of "
        "all sync events and MUST be included when the question asks about events or methods.\n"
        "  • All mixin/parent classes listed in the class definition\n"
        "Call all of these BEFORE calling read_file on anything. One batch, one round.\n\n"
        "STEP 1c — After reviewing outlines: call read_file on every method relevant to "
        "the question. For questions about 'what methods must subclasses implement' or "
        "'what does this class expose', read EVERY method in the outline that is either: "
        "(a) decorated @abstractmethod, (b) raises NotImplementedError, or (c) documented "
        "as an override point. Do not stop after finding the first abstract method.\n\n"
        "STEP 1d — If any class references other types (TypedDicts, parent classes, field "
        "types defined elsewhere), call get_class_outline or read_file on those too.\n\n"
        "STEP 2 — When answering, be exhaustive. Enumerate:\n"
        "- All fields / attributes and their types (read referenced TypedDicts/dataclasses for sub-fields)\n"
        "- ALL abstract or required methods subclasses must implement (check every @abstractmethod in outline)\n"
        "- Both sync and async method variants (e.g. invoke/ainvoke, on_*/async on_*)\n"
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
    model = ChatAnthropic(model=model_name, temperature=0).bind_tools([get_class_outline, read_file])
    system = SystemMessage(content=_build_system_prompt(state["retrieved_chunks"]))
    messages: list = [system] + list(state["messages"])

    response = None
    total_input_tokens = 0
    total_output_tokens = 0
    tool_trace: list[dict] = []

    def _add_usage(r):
        nonlocal total_input_tokens, total_output_tokens
        u = r.usage_metadata or {}
        total_input_tokens += u.get("input_tokens", 0)
        total_output_tokens += u.get("output_tokens", 0)

    budget_exhausted = False
    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            response = model.invoke(messages)
            _add_usage(response)
            if not response.tool_calls:
                break
            messages.append(response)
            for tc in response.tool_calls:
                fn = get_class_outline if tc["name"] == "get_class_outline" else read_file
                result = fn.invoke(tc["args"])
                tool_trace.append({"round": round_num + 1, "tool": tc["name"], "args": tc["args"]})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            budget_exhausted = True
            messages.append(HumanMessage(content=(
                "Tool budget exhausted. Write your final answer NOW using only the "
                "context already gathered. Do not request any more tools. "
                "Cite with [path:start-end] for every claim."
            )))
            model_no_tools = ChatAnthropic(model=model_name, temperature=0)
            response = model_no_tools.invoke(messages)
            _add_usage(response)
    except anthropic.APIError as e:
        return {"messages": list(state["messages"]) + [
            AIMessage(content=(
                "Anthropic API error — try again. "
                f"If it persists, check your API key and rate limits. ({e})"
            ))
        ]}

    content = response.content
    if not isinstance(content, str):
        content = "".join(
            (b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            for b in content
        )
    validated = validate_citations(content, _get_db())
    final = AIMessage(
        content=validated,
        usage_metadata={
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        additional_kwargs={"tool_trace": tool_trace, "budget_exhausted": budget_exhausted},
    )
    return {"messages": list(state["messages"]) + [final]}
