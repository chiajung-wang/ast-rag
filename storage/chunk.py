from __future__ import annotations
import hashlib
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    file_path: str
    symbol_name: str
    symbol_type: str       # "function" | "class" | "method"
    parent_class: str | None
    line_start: int
    line_end: int
    docstring: str | None
    text: str
    embed_text: str


def make_chunk(
    file_path: str,
    symbol_name: str,
    symbol_type: str,
    parent_class: str | None,
    line_start: int,
    line_end: int,
    docstring: str | None,
    text: str,
) -> Chunk:
    embed_text = f"{parent_class}.{symbol_name}: {text}" if parent_class else text
    chunk_id = hashlib.sha256(
        f"{file_path}{symbol_name}{line_start}{line_end}{text}".encode()
    ).hexdigest()
    return Chunk(
        id=chunk_id,
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        parent_class=parent_class,
        line_start=line_start,
        line_end=line_end,
        docstring=docstring,
        text=text,
        embed_text=embed_text,
    )
