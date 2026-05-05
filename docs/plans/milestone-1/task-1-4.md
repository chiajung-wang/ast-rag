# Task 1.4: Storage Layer

**What to build:** SQLite + sqlite-vec wrapper exposing all DB operations needed by the indexer and retrieval pipeline.

**Blocked by:** Task 1.2 (Chunk dataclass)

**Acceptance criteria:**
- [ ] `DB` initialises schema on first open (idempotent — safe to call twice)
- [ ] `insert_chunk` returns `rowid`, skips duplicates via `INSERT OR IGNORE`
- [ ] `insert_embedding` / `has_embedding` form the idempotency gate for the embedder
- [ ] `vector_search` returns `list[Chunk]` ordered by cosine distance
- [ ] `symbol_lookup` is case-insensitive exact match
- [ ] `chunk_exists_at` used by citation validator
- [ ] `all_chunks` / `all_symbol_names` used by BM25 index + retrieve heuristic
- [ ] All tests pass against a real (temp-file) SQLite DB

---

**Files:**
- Create: `storage/db.py`
- Create: `tests/test_db.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_db.py`:
```python
import struct
import tempfile
from pathlib import Path
import pytest
from storage.chunk import make_chunk
from storage.db import DB

EMBEDDING_DIM = 1536


def _fake_embedding(seed: float = 0.1) -> list[float]:
    return [seed] * EMBEDDING_DIM


def _serialize(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


@pytest.fixture
def db(tmp_path):
    return DB(str(tmp_path / "test.db"))


@pytest.fixture
def chunk_a():
    return make_chunk("runnables/base.py", "RunnableSequence", "class", None, 10, 100, "A runnable.", "class RunnableSequence:\n    pass")


@pytest.fixture
def chunk_b():
    return make_chunk("runnables/base.py", "invoke", "method", "RunnableSequence", 20, 40, None, "def invoke(self): pass")


def test_insert_chunk_returns_rowid(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    assert isinstance(rowid, int)
    assert rowid > 0


def test_insert_chunk_duplicate_returns_same_rowid(db, chunk_a):
    r1 = db.insert_chunk(chunk_a)
    r2 = db.insert_chunk(chunk_a)
    assert r1 == r2


def test_has_embedding_false_before_insert(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    assert db.has_embedding(rowid) is False


def test_has_embedding_true_after_insert(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    db.insert_embedding(rowid, _fake_embedding())
    assert db.has_embedding(rowid) is True


def test_vector_search_returns_chunks(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    db.insert_embedding(rowid, _fake_embedding(0.1))
    results = db.vector_search(_fake_embedding(0.1), k=5)
    assert len(results) == 1
    assert results[0].symbol_name == "RunnableSequence"


def test_symbol_lookup_exact(db, chunk_a):
    db.insert_chunk(chunk_a)
    result = db.symbol_lookup("RunnableSequence")
    assert result is not None
    assert result.symbol_name == "RunnableSequence"


def test_symbol_lookup_case_insensitive(db, chunk_a):
    db.insert_chunk(chunk_a)
    assert db.symbol_lookup("runnablesequence") is not None
    assert db.symbol_lookup("RUNNABLESEQUENCE") is not None


def test_symbol_lookup_miss_returns_none(db):
    assert db.symbol_lookup("DoesNotExist") is None


def test_chunk_exists_at_hit(db, chunk_a):
    db.insert_chunk(chunk_a)
    # line_start=10, line_end=100 — query range entirely inside
    assert db.chunk_exists_at("runnables/base.py", 10, 100) is True


def test_chunk_exists_at_miss_wrong_file(db, chunk_a):
    db.insert_chunk(chunk_a)
    assert db.chunk_exists_at("other/file.py", 10, 100) is False


def test_all_chunks_returns_all(db, chunk_a, chunk_b):
    db.insert_chunk(chunk_a)
    db.insert_chunk(chunk_b)
    chunks = db.all_chunks()
    names = {c.symbol_name for c in chunks}
    assert "RunnableSequence" in names
    assert "invoke" in names


def test_all_symbol_names(db, chunk_a, chunk_b):
    db.insert_chunk(chunk_a)
    db.insert_chunk(chunk_b)
    names = db.all_symbol_names()
    assert "RunnableSequence" in names
    assert "invoke" in names


def test_schema_init_is_idempotent(tmp_path):
    path = str(tmp_path / "test.db")
    DB(path)  # first open
    DB(path)  # second open — must not raise
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_db.py -v
```

Expected: `ModuleNotFoundError: No module named 'storage.db'`

- [ ] **Step 3: Write `storage/db.py`**

```python
from __future__ import annotations
import sqlite3
import struct
from storage.chunk import Chunk, make_chunk

EMBEDDING_DIM = 1536


def _serialize(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class DB:
    def __init__(self, path: str = "index.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                parent_class TEXT,
                line_start  INTEGER NOT NULL,
                line_end    INTEGER NOT NULL,
                docstring   TEXT,
                text        TEXT NOT NULL,
                embed_text  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_symbol
                ON chunks(lower(symbol_name));
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding FLOAT[{EMBEDDING_DIM}]
            );
        """)
        self.conn.commit()

    # ── chunk writes ──────────────────────────────────────────────────────────

    def insert_chunk(self, chunk: Chunk) -> int:
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO chunks
                (id, file_path, symbol_name, symbol_type, parent_class,
                 line_start, line_end, docstring, text, embed_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id, chunk.file_path, chunk.symbol_name, chunk.symbol_type,
                chunk.parent_class, chunk.line_start, chunk.line_end,
                chunk.docstring, chunk.text, chunk.embed_text,
            ),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (chunk.id,)
            ).fetchone()
            return row["rowid"]
        return cur.lastrowid

    # ── embedding writes / reads ──────────────────────────────────────────────

    def has_embedding(self, rowid: int) -> bool:
        return self.conn.execute(
            "SELECT rowid FROM vec_chunks WHERE rowid = ?", (rowid,)
        ).fetchone() is not None

    def insert_embedding(self, rowid: int, embedding: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize(embedding)),
        )
        self.conn.commit()

    # ── queries ───────────────────────────────────────────────────────────────

    def vector_search(self, embedding: list[float], k: int = 10) -> list[Chunk]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.file_path, c.symbol_name, c.symbol_type, c.parent_class,
                   c.line_start, c.line_end, c.docstring, c.text, c.embed_text
            FROM vec_chunks v
            JOIN chunks c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_serialize(embedding), k),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def symbol_lookup(self, name: str) -> Chunk | None:
        row = self.conn.execute(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text
            FROM chunks WHERE lower(symbol_name) = lower(?) LIMIT 1
            """,
            (name,),
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def chunk_exists_at(self, file_path: str, line_start: int, line_end: int) -> bool:
        return self.conn.execute(
            """
            SELECT 1 FROM chunks
            WHERE file_path = ? AND line_start <= ? AND line_end >= ?
            LIMIT 1
            """,
            (file_path, line_start, line_end),
        ).fetchone() is not None

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text
            FROM chunks
            """
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def all_symbol_names(self) -> set[str]:
        rows = self.conn.execute("SELECT symbol_name FROM chunks").fetchall()
        return {r["symbol_name"] for r in rows}


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        file_path=row["file_path"],
        symbol_name=row["symbol_name"],
        symbol_type=row["symbol_type"],
        parent_class=row["parent_class"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        docstring=row["docstring"],
        text=row["text"],
        embed_text=row["embed_text"],
    )
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
pytest tests/test_db.py -v
```

Expected: 13 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add storage/db.py tests/test_db.py
git commit -m "feat: DB storage layer — SQLite + sqlite-vec CRUD and vector search"
```
