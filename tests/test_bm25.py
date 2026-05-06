from storage.chunk import make_chunk
from retrieval.bm25_index import BM25Index, _tokenize


def _make_chunk(symbol_name: str, symbol_type: str = "function", parent_class: str | None = None):
    text = f"def {symbol_name}(): pass"
    return make_chunk("mod.py", symbol_name, symbol_type, parent_class, 1, 2, None, text)


# ── tokenizer ─────────────────────────────────────────────────────────────────

def test_tokenize_camel_case():
    tokens = _tokenize("RunnableSequence")
    assert "runnable" in tokens
    assert "sequence" in tokens
    assert "runnablesequence" in tokens


def test_tokenize_snake_case():
    tokens = _tokenize("invoke_async")
    assert "invoke" in tokens
    assert "async" in tokens
    assert "invoke_async" in tokens


def test_tokenize_plain_word():
    tokens = _tokenize("runnable")
    assert "runnable" in tokens


def test_tokenize_mixed():
    tokens = _tokenize("BaseLLM.invoke_async")
    assert "invoke" in tokens
    assert "async" in tokens
    assert "invoke_async" in tokens


# ── BM25Index ─────────────────────────────────────────────────────────────────

def test_bm25_search_returns_matching_chunk():
    chunks = [
        _make_chunk("RunnableSequence", "class"),
        _make_chunk("BaseLLM", "class"),
    ]
    idx = BM25Index(chunks)
    results = idx.search("RunnableSequence", k=5)
    names = [c.symbol_name for c in results]
    assert "RunnableSequence" in names


def test_bm25_search_top_result_is_most_relevant():
    chunks = [
        _make_chunk("RunnableSequence", "class"),
        _make_chunk("BaseLLM", "class"),
        _make_chunk("RunnableParallel", "class"),
    ]
    idx = BM25Index(chunks)
    results = idx.search("RunnableSequence", k=3)
    assert results[0].symbol_name == "RunnableSequence"


def test_bm25_search_k_limits_results():
    chunks = [_make_chunk(f"Fn{i}") for i in range(10)]
    idx = BM25Index(chunks)
    results = idx.search("Fn", k=3)
    assert len(results) <= 3


def test_bm25_search_k_larger_than_corpus():
    chunks = [_make_chunk("OnlyOne")]
    idx = BM25Index(chunks)
    results = idx.search("OnlyOne", k=100)
    assert len(results) == 1


def test_bm25_empty_corpus_returns_empty():
    idx = BM25Index([])
    assert idx.search("anything") == []


def test_bm25_from_db(tmp_path):
    from storage.db import DB
    db = DB(str(tmp_path / "test.db"))
    chunk = _make_chunk("RunnableSequence", "class")
    db.insert_chunk(chunk)
    idx = BM25Index.from_db(db)
    results = idx.search("RunnableSequence", k=5)
    assert any(c.symbol_name == "RunnableSequence" for c in results)
