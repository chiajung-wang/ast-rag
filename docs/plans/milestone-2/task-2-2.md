# Task 2.2: RRF Fusion

**What to build:** Reciprocal Rank Fusion function that merges BM25 top-10 and dense top-10 result lists into a unified top-N ranking. Pure function, no I/O.

**Blocked by:** Task 1.2 (Chunk dataclass)

**Acceptance criteria:**
- [ ] Chunk appearing in both BM25 and dense lists scores higher than a chunk in only one list
- [ ] Chunk IDs are deduplicated — same chunk appears at most once in output
- [ ] `top_n` caps output length
- [ ] Single-list hits are included (not filtered out)
- [ ] Empty inputs return `[]`
- [ ] All tests pass

---

**Files:**
- Create: `retrieval/rrf.py`
- Create: `tests/test_rrf.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_rrf.py`:

```python
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
    # rank-1 BM25 + rank-1 dense beats rank-1 BM25 + rank-10 dense
    high = _chunk("high")
    low = _chunk("low")
    other = _chunk("other")
    results = rrf(
        [high, low],       # high=rank1, low=rank2 in BM25
        [high, other],     # high=rank1 in dense
        top_n=3,
    )
    assert results[0].symbol_name == "high"
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_rrf.py -v
```

Expected: `ModuleNotFoundError: No module named 'retrieval.rrf'`

- [ ] **Step 3: Write `retrieval/rrf.py`**

```python
from __future__ import annotations
from storage.chunk import Chunk


def rrf(
    bm25_results: list[Chunk],
    dense_results: list[Chunk],
    k: int = 60,
    top_n: int = 5,
) -> list[Chunk]:
    scores: dict[str, float] = {}
    id_to_chunk: dict[str, Chunk] = {}

    for rank, chunk in enumerate(bm25_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        id_to_chunk[chunk.id] = chunk

    for rank, chunk in enumerate(dense_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        id_to_chunk[chunk.id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [id_to_chunk[chunk_id] for chunk_id, _ in ranked[:top_n]]
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_rrf.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add retrieval/rrf.py tests/test_rrf.py
git commit -m "feat(retrieval): RRF fusion — merge BM25 and dense results"
```
