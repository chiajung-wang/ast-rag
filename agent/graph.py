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
