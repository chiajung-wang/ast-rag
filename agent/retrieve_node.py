from __future__ import annotations
from agent.state import AgentState
from retrieval.pipeline import retrieve


def retrieve_node(state: AgentState) -> dict:
    query = state["messages"][-1].content
    chunks = retrieve(query)
    return {"retrieved_chunks": chunks}
