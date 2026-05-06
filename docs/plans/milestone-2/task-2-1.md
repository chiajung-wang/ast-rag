# Task 2.1: BM25 Index

**What to build:** In-memory BM25 index over all chunk `embed_text` values. Tokenizer expands camelCase and snake_case symmetrically. Index built once from DB at startup and cached as a module-level singleton in the pipeline.

**Blocked by:** Task 1.4 (DB layer)

**Acceptance criteria:**
- [ ] `_tokenize("RunnableSequence")` → includes `"runnable"`, `"sequence"`, `"runnablesequence"`
- [ ] `_tokenize("invoke_async")` → includes `"invoke"`, `"async"`, `"invoke_async"`
- [ ] `BM25Index.search("RunnableSequence")` returns chunk whose `symbol_name == "RunnableSequence"` in top results
- [ ] `BM25Index.search(...)` with `k` larger than corpus returns all scored chunks (no error)
- [ ] Empty corpus → `search` returns `[]`
- [ ] All tests pass

---

**Files:**
- Create: `retrieval/bm25_index.py`
- Create: `tests/test_bm25.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_bm25.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_bm25.py -v
```

Expected: `ModuleNotFoundError: No module named 'retrieval.bm25_index'`

- [ ] **Step 3: Write `retrieval/bm25_index.py`**

```python
from __future__ import annotations
import re
from rank_bm25 import BM25Okapi
from storage.chunk import Chunk


def _tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r'[A-Za-z0-9_]+', text):
        snake_parts = [p for p in word.split('_') if p]
        if len(snake_parts) > 1:
            tokens.extend(p.lower() for p in snake_parts)
            tokens.append(word.lower())
        else:
            s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', word)
            s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
            parts = s.split()
            if len(parts) > 1:
                tokens.extend(p.lower() for p in parts)
                tokens.append(word.lower())
            else:
                tokens.append(word.lower())
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        tokenized = [_tokenize(c.embed_text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, k: int = 10) -> list[Chunk]:
        if not self._chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top if scores[i] > 0]

    @classmethod
    def from_db(cls, db) -> "BM25Index":
        return cls(db.all_chunks())
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_bm25.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add retrieval/bm25_index.py tests/test_bm25.py
git commit -m "feat(retrieval): BM25 index with camelCase+snake_case tokenizer"
```
