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


def test_symbol_lookup_prefers_class_over_method_and_function(db):
    """1296 names cover 2414 chunks — collisions need a stable tie-break."""
    db.insert_chunk(make_chunk("z/last.py", "Tee", "function", None, 1, 5, None, "def Tee(): pass"))
    db.insert_chunk(make_chunk("m/mid.py", "Tee", "method", "Runnable", 1, 5, None, "def Tee(self): pass"))
    db.insert_chunk(make_chunk("a/first.py", "Tee", "class", None, 1, 5, None, "class Tee: pass"))

    result = db.symbol_lookup("Tee")
    assert result.symbol_type == "class"
    assert result.file_path == "a/first.py"


def test_symbol_lookup_tie_break_orders_by_file_path(db):
    """Same symbol_type in two files — lowest file_path wins, deterministically."""
    db.insert_chunk(make_chunk("z/last.py", "ToolCall", "class", None, 1, 5, None, "class ToolCall: pass"))
    db.insert_chunk(make_chunk("a/first.py", "ToolCall", "class", None, 1, 5, None, "class ToolCall: ..."))

    assert db.symbol_lookup("ToolCall").file_path == "a/first.py"


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
