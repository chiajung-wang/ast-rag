from __future__ import annotations
import re
from indexer.corpus_config import COMMIT_SHA, CORPUS_SUBPATH, REPO_URL

_CITATION_RE = re.compile(r'\[([^:\]\s]+):(\d+)-(\d+)\]')


def parse_citations(text: str) -> list[tuple[str, int, int]]:
    return [(p, int(s), int(e)) for p, s, e in _CITATION_RE.findall(text)]


def build_permalink(path: str, line_start: int, line_end: int) -> str:
    # CORPUS_SUBPATH, not a hardcoded prefix: chunk file_path is relative to
    # the corpus root, so a literal "libs/core/" drops the langchain_core/
    # segment and every link 404s.
    return (
        f"{REPO_URL}/blob/{COMMIT_SHA}"
        f"/{CORPUS_SUBPATH}/{path}#L{line_start}-L{line_end}"
    )
