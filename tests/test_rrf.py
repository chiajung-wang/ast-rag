from storage.chunk import make_chunk
from retrieval.rrf import rrf


def _chunk(symbol_name: str) -> object:
    return make_chunk("f.py", symbol_name, "function", None, 1, 2, None, f"def {symbol_name}(): pass")


def test_rrf_chunk_in_both_lists_ranks_first():
    shared = _chunk("shared")
    bm25_only = _chunk("bm25_only")
    dense_only = _chunk("dense_only")

    results = rrf([shared, bm25_only], [shared, dense_only], top_n=3)
    assert results[0].symbol_name == "shared"


def test_rrf_deduplicates_shared_chunk():
    shared = _chunk("shared")
    results = rrf([shared], [shared], top_n=5)
    assert len(results) == 1


def test_rrf_top_n_limits_output():
    chunks = [_chunk(f"c{i}") for i in range(10)]
    results = rrf(chunks[:5], chunks[5:], top_n=3)
    assert len(results) == 3


def test_rrf_single_list_hits_included():
    a = _chunk("a")
    b = _chunk("b")
    results = rrf([a], [b], top_n=5)
    names = {c.symbol_name for c in results}
    assert "a" in names
    assert "b" in names


def test_rrf_empty_bm25():
    chunk = _chunk("only_dense")
    results = rrf([], [chunk], top_n=5)
    assert len(results) == 1
    assert results[0].symbol_name == "only_dense"


def test_rrf_empty_dense():
    chunk = _chunk("only_bm25")
    results = rrf([chunk], [], top_n=5)
    assert results[0].symbol_name == "only_bm25"


def test_rrf_both_empty():
    assert rrf([], [], top_n=5) == []


def test_rrf_rank_order_respected():
    high = _chunk("high")
    low = _chunk("low")
    other = _chunk("other")
    results = rrf(
        [high, low],
        [high, other],
        top_n=3,
    )
    assert results[0].symbol_name == "high"
