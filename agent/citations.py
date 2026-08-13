from __future__ import annotations
import re
from dataclasses import dataclass
from storage.db import DB
from indexer.corpus_config import CLONE_DIR, CORPUS_SUBPATH


@dataclass
class CitationStats:
    emitted: int              # markers found before validation
    stripped: int              # markers removed (bad range or unknown path)
    hallucinated_paths: int    # of those stripped, how many name a path never indexed at all
    precise: int                # of markers that survived, how many name a symbol the answer text mentions
    survived: int               # markers that survived validation (emitted - stripped)

    @property
    def strip_rate(self) -> float:
        return self.stripped / self.emitted if self.emitted else 0.0

    @property
    def hallucinated_path_rate(self) -> float:
        return self.hallucinated_paths / self.emitted if self.emitted else 0.0

    @property
    def precision(self) -> float:
        # Precision is defined over surviving markers, not all emitted ones --
        # it measures whether a *validated* citation is actually about the
        # thing named at that location, not whether validation itself worked
        # (that's strip_rate's job).
        return self.precise / self.survived if self.survived else 0.0

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

    Thin wrapper over validate_citations_with_stats that keeps the original
    signature working for the one existing caller that doesn't need the
    stats.
    """
    result, _stats = validate_citations_with_stats(text, db)
    return result


def validate_citations_with_stats(text: str, db: DB) -> tuple[str, CitationStats]:
    """Same behavior as validate_citations, plus the metrics task 6.7 wants:
    strip rate, hallucinated-path rate, and precision.

    Precision here answers a narrower question than "is this citation
    correct" -- chunk_exists_at (containment) already establishes that a
    surviving marker's range sits inside some indexed symbol. Precision
    checks whether that symbol is one the answer actually talks about: for
    every chunk overlapping the marker's range, does its symbol_name appear
    as a substring of the answer text? A class-level citation attached to an
    unrelated method's claim would pass containment but fail this -- that's
    the gap between "the range is real" and "the range is about the right
    thing", which is exactly what containment alone can't catch.
    """
    matches = _CITATION_RE.findall(text)
    if not matches:
        return text, CitationStats(emitted=0, stripped=0, hallucinated_paths=0, precise=0, survived=0)

    result = text
    invalid: list[str] = []
    hallucinated_paths = 0
    precise = 0
    survived = 0
    for path, start, end in matches:
        marker = f"[{path}:{start}-{end}]"
        canonical = normalize_path(path)
        start_i, end_i = int(start), int(end)
        chunks = db.chunks_at(canonical, start_i, end_i)
        if chunks:
            survived += 1
            if any(c.symbol_name in text for c in chunks):
                precise += 1
            if canonical != path:
                result = result.replace(marker, f"[{canonical}:{start_i}-{end_i}]")
        else:
            invalid.append(marker)
            if not db.file_path_known(canonical):
                hallucinated_paths += 1

    stats = CitationStats(
        emitted=len(matches), stripped=len(invalid),
        hallucinated_paths=hallucinated_paths, precise=precise, survived=survived,
    )

    if not invalid:
        return result, stats

    for marker in invalid:
        result = result.replace(marker, "")

    n = len(invalid)
    result = result.rstrip() + f"\n\n*{n} citation(s) could not be verified and were removed.*"
    return result, stats
