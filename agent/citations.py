from __future__ import annotations
import re
from storage.db import DB

_CITATION_RE = re.compile(r'\[([^:\]\s]+):(\d+)-(\d+)\]')


def validate_citations(text: str, db: DB) -> str:
    matches = _CITATION_RE.findall(text)
    if not matches:
        return text

    invalid: list[str] = []
    for path, start, end in matches:
        if not db.chunk_exists_at(path, int(start), int(end)):
            invalid.append(f"[{path}:{start}-{end}]")

    if not invalid:
        return text

    result = text
    for marker in invalid:
        result = result.replace(marker, "")

    n = len(invalid)
    result = result.rstrip() + f"\n\n*{n} citation(s) could not be verified and were removed.*"
    return result
