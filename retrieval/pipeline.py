from __future__ import annotations
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from storage.chunk import Chunk
from storage.db import DB
from retrieval.bm25_index import BM25Index
from retrieval.rrf import rrf
from indexer.corpus_config import CLONE_DIR, CORPUS_SUBPATH, DB_PATH

load_dotenv()

CORPUS_ROOT = Path(CLONE_DIR) / CORPUS_SUBPATH

_db: DB | None = None
_bm25: BM25Index | None = None
_client: OpenAI | None = None

_SYMBOL_RE = re.compile(
    r'\b([A-Z][a-zA-Z0-9]{2,}|[a-z][a-z0-9]*(?:_[a-z0-9]+)+|[A-Z]{2,})\b'
)


def _get_db() -> DB:
    global _db
    if _db is None:
        _db = DB(DB_PATH)
    return _db


def _get_bm25() -> BM25Index:
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index.from_db(_get_db())
    return _bm25


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _embed_query(query: str) -> list[float]:
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=[query[:24_000]],
    )
    return response.data[0].embedding


def search_corpus(query: str, k: int = 5) -> list[Chunk]:
    bm25_results = _get_bm25().search(query, k=10)
    embedding = _embed_query(query)
    dense_results = _get_db().vector_search(embedding, k=10)
    return rrf(bm25_results, dense_results, top_n=k)


def find_symbol(name: str) -> Chunk | None:
    return _get_db().symbol_lookup(name)


def read_file(path: str, line_start: int, line_end: int) -> str:
    MAX_LINES = 100
    full_path = CORPUS_ROOT / path
    try:
        lines = full_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return f"[error: file not found: {path}]"
    requested = line_end - line_start + 1
    actual_end = min(line_end, line_start + MAX_LINES - 1)
    result = "\n".join(lines[line_start - 1 : actual_end])
    if requested > MAX_LINES:
        result += f"\n[truncated: requested {requested} lines, returned {MAX_LINES}]"
    return result


def retrieve(query: str, k: int = 5) -> list[Chunk]:
    symbol_names = _get_db().all_symbol_names()
    candidates = {m.group(1) for m in _SYMBOL_RE.finditer(query)}
    lower_map = {s.lower(): s for s in symbol_names}

    results: list[Chunk] = []
    seen: set[str] = set()

    for candidate in candidates:
        canonical = lower_map.get(candidate.lower())
        if canonical:
            chunk = find_symbol(canonical)
            if chunk and chunk.id not in seen:
                results.append(chunk)
                seen.add(chunk.id)
            break

    for c in search_corpus(query, k=k):
        if c.id not in seen and len(results) < k:
            results.append(c)
            seen.add(c.id)

    return results[:k]
