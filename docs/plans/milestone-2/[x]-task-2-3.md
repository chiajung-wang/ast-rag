# Task 2.3: Retrieval Pipeline

**What to build:** `retrieval/pipeline.py` exposing `search_corpus`, `find_symbol`, `read_file`, and `retrieve` as plain Python functions. Module-level singletons for DB and BM25Index; OpenAI embedding for query encoding. `retrieve` includes heuristic symbol pre-check.

**Blocked by:** Task 2.1 (BM25 index), Task 2.2 (RRF)

**Acceptance criteria:**
- [x] `find_symbol("RunnableSequence")` returns the class chunk (case-insensitive)
- [x] `find_symbol("doesnotexist")` returns `None`
- [x] `read_file("runnables/base.py", 1, 10)` returns 10 lines
- [x] `read_file("runnables/base.py", 1, 500)` returns exactly 100 lines + truncation notice
- [x] `read_file("nonexistent.py", 1, 5)` returns `[error: ...]` string (no exception)
- [x] `search_corpus("RunnableSequence")` returns ≥1 result without hitting real OpenAI
- [x] `retrieve("RunnableSequence", k=5)` puts the `find_symbol` hit first
- [x] All tests pass

---

**Files:**
- Create: `retrieval/pipeline.py`
- Create: `tests/test_pipeline.py`

---

- [x] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
from storage.chunk import make_chunk
from storage.db import DB


def _make_chunk(symbol_name: str, symbol_type: str = "function", parent_class: str | None = None):
    text = f"def {symbol_name}(): pass"
    return make_chunk("runnables/base.py", symbol_name, symbol_type, parent_class, 1, 5, None, text)


@pytest.fixture
def db(tmp_path):
    return DB(str(tmp_path / "test.db"))


@pytest.fixture
def populated_db(db):
    db.insert_chunk(_make_chunk("RunnableSequence", "class"))
    db.insert_chunk(_make_chunk("invoke", "method", "RunnableSequence"))
    return db


# ── find_symbol ───────────────────────────────────────────────────────────────

def test_find_symbol_hit(populated_db):
    from retrieval.pipeline import find_symbol
    with patch("retrieval.pipeline._get_db", return_value=populated_db):
        result = find_symbol("RunnableSequence")
    assert result is not None
    assert result.symbol_name == "RunnableSequence"


def test_find_symbol_case_insensitive(populated_db):
    from retrieval.pipeline import find_symbol
    with patch("retrieval.pipeline._get_db", return_value=populated_db):
        assert find_symbol("runnablesequence") is not None


def test_find_symbol_miss(populated_db):
    from retrieval.pipeline import find_symbol
    with patch("retrieval.pipeline._get_db", return_value=populated_db):
        assert find_symbol("DoesNotExist") is None


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file_returns_correct_lines(tmp_path):
    f = tmp_path / "base.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 21)))

    from retrieval.pipeline import read_file
    with patch("retrieval.pipeline.CORPUS_ROOT", tmp_path):
        result = read_file("base.py", 1, 5)

    assert result == "line1\nline2\nline3\nline4\nline5"


def test_read_file_truncates_at_100_lines(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 301)))

    from retrieval.pipeline import read_file
    with patch("retrieval.pipeline.CORPUS_ROOT", tmp_path):
        result = read_file("big.py", 1, 300)

    lines = result.split("\n")
    assert lines[-1].startswith("[truncated:")
    assert "returned 100" in lines[-1]
    assert len(lines) == 101  # 100 content lines + truncation notice


def test_read_file_missing_returns_error(tmp_path):
    from retrieval.pipeline import read_file
    with patch("retrieval.pipeline.CORPUS_ROOT", tmp_path):
        result = read_file("nonexistent.py", 1, 5)
    assert result.startswith("[error:")


# ── search_corpus ─────────────────────────────────────────────────────────────

def test_search_corpus_returns_results(populated_db, tmp_path):
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import search_corpus

    bm25 = BM25Index.from_db(populated_db)
    fake_embedding = [0.1] * 1536
    rowid = populated_db.insert_chunk(_make_chunk("RunnableSequence", "class"))
    populated_db.insert_embedding(rowid, fake_embedding)

    with patch("retrieval.pipeline._get_db", return_value=populated_db), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        results = search_corpus("RunnableSequence", k=5)

    assert len(results) >= 1


# ── retrieve ──────────────────────────────────────────────────────────────────

def test_retrieve_symbol_match_appears_first(populated_db):
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import retrieve

    bm25 = BM25Index.from_db(populated_db)
    fake_embedding = [0.1] * 1536
    rowid = populated_db.insert_chunk(_make_chunk("RunnableSequence", "class"))
    populated_db.insert_embedding(rowid, fake_embedding)

    with patch("retrieval.pipeline._get_db", return_value=populated_db), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        results = retrieve("RunnableSequence", k=5)

    assert results[0].symbol_name == "RunnableSequence"


def test_retrieve_no_symbol_falls_back_to_search(populated_db):
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import retrieve

    bm25 = BM25Index.from_db(populated_db)
    fake_embedding = [0.1] * 1536
    rowid = populated_db.insert_chunk(_make_chunk("RunnableSequence", "class"))
    populated_db.insert_embedding(rowid, fake_embedding)

    with patch("retrieval.pipeline._get_db", return_value=populated_db), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        results = retrieve("how does streaming work", k=5)

    assert isinstance(results, list)
```

- [x] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'retrieval.pipeline'`

- [x] **Step 3: Write `retrieval/pipeline.py`**

```python
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

    remaining = k - len(results)
    if remaining > 0:
        for c in search_corpus(query, k=k):
            if c.id not in seen and len(results) < k:
                results.append(c)
                seen.add(c.id)

    return results[:k]
```

- [x] **Step 4: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_pipeline.py -v
```

Expected: all tests PASSED.

- [x] **Step 5: Commit**

```bash
git add retrieval/pipeline.py tests/test_pipeline.py
git commit -m "feat(retrieval): pipeline — search_corpus, find_symbol, read_file, retrieve"
```
