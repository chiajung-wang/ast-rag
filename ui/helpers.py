from __future__ import annotations
import re
from indexer.corpus_config import COMMIT_SHA

_CITATION_RE = re.compile(r'\[([^:\]\s]+):(\d+)-(\d+)\]')


def parse_citations(text: str) -> list[tuple[str, int, int]]:
    return [(p, int(s), int(e)) for p, s, e in _CITATION_RE.findall(text)]


def build_permalink(path: str, line_start: int, line_end: int) -> str:
    return (
        f"https://github.com/langchain-ai/langchain/blob/{COMMIT_SHA}"
        f"/libs/core/{path}#L{line_start}-L{line_end}"
    )
