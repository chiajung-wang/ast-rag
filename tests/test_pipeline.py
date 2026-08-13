from unittest.mock import patch
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


def test_read_file_clamps_line_start_below_one(tmp_path):
    """line_start=0 used to slice lines[-1:] and silently return the last line."""
    f = tmp_path / "base.py"
    f.write_text("\n".join(f"line{i}" for i in range(1, 21)))

    from retrieval.pipeline import read_file
    with patch("retrieval.pipeline.CORPUS_ROOT", tmp_path):
        result = read_file("base.py", 0, 3)

    assert result == "line1\nline2\nline3"


def test_read_file_missing_returns_error(tmp_path):
    from retrieval.pipeline import read_file
    with patch("retrieval.pipeline.CORPUS_ROOT", tmp_path):
        result = read_file("nonexistent.py", 1, 5)
    assert result.startswith("[error:")


# ── search_corpus ─────────────────────────────────────────────────────────────

def test_search_corpus_returns_results(populated_db):
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import search_corpus

    bm25 = BM25Index.from_db(populated_db)
    fake_embedding = [0.1] * 1536
    chunk = _make_chunk("RunnableSequence", "class")
    rowid = populated_db.insert_chunk(chunk)
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
    chunk = _make_chunk("RunnableSequence", "class")
    rowid = populated_db.insert_chunk(chunk)
    populated_db.insert_embedding(rowid, fake_embedding)

    with patch("retrieval.pipeline._get_db", return_value=populated_db), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        results = retrieve("RunnableSequence", k=5)

    assert results[0].symbol_name == "RunnableSequence"


def _two_symbol_db(db):
    """Two classes, both embedded, so retrieve() can reach either one."""
    fake_embedding = [0.1] * 1536
    for name in ("AlphaRunnable", "BetaRunnable"):
        chunk = make_chunk(f"{name.lower()}.py", name, "class", None, 1, 5, None, f"class {name}: pass")
        rowid = db.insert_chunk(chunk)
        db.insert_embedding(rowid, fake_embedding)
    return db, fake_embedding


def test_retrieve_picks_first_symbol_in_query_order(db):
    """candidates was a set — hash randomization changed the pick between runs."""
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import retrieve

    populated, fake_embedding = _two_symbol_db(db)
    bm25 = BM25Index.from_db(populated)

    with patch("retrieval.pipeline._get_db", return_value=populated), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        forward = retrieve("compare BetaRunnable with AlphaRunnable", k=5)
        reversed_ = retrieve("compare AlphaRunnable with BetaRunnable", k=5)

    assert forward[0].symbol_name == "BetaRunnable"
    assert reversed_[0].symbol_name == "AlphaRunnable"


def test_retrieve_no_symbol_falls_back_to_search(populated_db):
    from retrieval.bm25_index import BM25Index
    from retrieval.pipeline import retrieve

    bm25 = BM25Index.from_db(populated_db)
    fake_embedding = [0.1] * 1536
    chunk = _make_chunk("RunnableSequence", "class")
    rowid = populated_db.insert_chunk(chunk)
    populated_db.insert_embedding(rowid, fake_embedding)

    with patch("retrieval.pipeline._get_db", return_value=populated_db), \
         patch("retrieval.pipeline._get_bm25", return_value=bm25), \
         patch("retrieval.pipeline._embed_query", return_value=fake_embedding):
        results = retrieve("how does streaming work", k=5)

    assert isinstance(results, list)
