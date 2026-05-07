from __future__ import annotations
from typing import TypedDict
from langchain_core.messages import BaseMessage
from storage.chunk import Chunk


class AgentState(TypedDict):
    messages: list[BaseMessage]
    retrieved_chunks: list[Chunk]
