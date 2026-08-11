from __future__ import annotations
import re
from storage.db import DB
from indexer.corpus_config import CLONE_DIR, CORPUS_SUBPATH

_CITATION_RE = re.compile(r'\[([^:\]\s]+):(\d+)-(\d+)\]')

# Chunk file_path is relative to the corpus root. The model often writes a
# longer path ("langchain_core/runnables/base.py"). Those citations are
# correct, so strip any known leading segment before the lookup instead of
# discarding the marker. Longest prefix first.
_PATH_PREFIXES = [
    "/".join(f"{CLONE_DIR}/{CORPUS_SUBPATH}".split("/")[i:]) + "/"
    for i in range(len(f"{CLONE_DIR}/{CORPUS_SUBPATH}".split("/")))
]


def normalize_path(path: str) -> str:
    """Strip a corpus-root prefix so the path matches a stored file_path."""
    cleaned = path.lstrip("/")
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    for prefix in _PATH_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):]
    return cleaned


def validate_citations(text: str, db: DB) -> str:
    """Drop unverifiable markers. Rewrite the survivors to canonical paths.

    Canonicalizing here keeps one path form downstream. The UI expander and
    the GitHub permalink both join file_path onto the corpus root, so a
    marker left in long form would resolve to a doubled path.
    """
    matches = _CITATION_RE.findall(text)
    if not matches:
        return text

    result = text
    invalid: list[str] = []
    for path, start, end in matches:
        marker = f"[{path}:{start}-{end}]"
        canonical = normalize_path(path)
        if db.chunk_exists_at(canonical, int(start), int(end)):
            if canonical != path:
                result = result.replace(marker, f"[{canonical}:{start}-{end}]")
        else:
            invalid.append(marker)

    if not invalid:
        return result

    for marker in invalid:
        result = result.replace(marker, "")

    n = len(invalid)
    result = result.rstrip() + f"\n\n*{n} citation(s) could not be verified and were removed.*"
    return result
