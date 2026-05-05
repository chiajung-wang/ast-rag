# Task 1.5: Embedder

**What to build:** OpenAI embedding wrapper that batches chunks, calls `text-embedding-3-small`, and writes to `DB` — skipping chunks whose embedding is already present (idempotency gate: `DB.has_embedding`).

**Blocked by:** Task 1.2 (Chunk), Task 1.4 (DB)

**Acceptance criteria:**
- [ ] `embed_chunks(chunks, db)` skips chunks where `db.has_embedding(rowid)` is True
- [ ] Calls OpenAI in batches of ≤100 chunks (avoids rate limits)
- [ ] Inserts embedding via `db.insert_embedding` after each batch
- [ ] Test verifies skip logic without making real API calls (mock)
- [ ] Test verifies batch size respected

---

**Files:**
- Create: `indexer/embedder.py`
- Create: `tests/test_embedder.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_embedder.py`:
```python
from unittest.mock import MagicMock, patch, call
import tempfile
from storage.chunk import make_chunk
from storage.db import DB
from indexer.embedder import embed_chunks, BATCH_SIZE


def _make_chunks(n: int) -> list:
    return [
        make_chunk(f"file_{i}.py", f"fn_{i}", "function", None, i, i + 5, None, f"def fn_{i}(): pass")
        for i in range(n)
    ]


def _fake_embedding(dim: int = 1536) -> list[float]:
    return [0.1] * dim


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_inserts_embeddings(mock_embed, tmp_path):
    mock_embed.return_value = [[0.1] * 1536]
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(1)
    rowid = db.insert_chunk(chunks[0])
    embed_chunks(chunks, db)
    assert db.has_embedding(rowid)


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_skips_existing(mock_embed, tmp_path):
    mock_embed.return_value = [[0.1] * 1536]
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(1)
    rowid = db.insert_chunk(chunks[0])
    db.insert_embedding(rowid, [0.1] * 1536)  # pre-insert

    embed_chunks(chunks, db)
    mock_embed.assert_not_called()


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_batches(mock_embed, tmp_path):
    n = BATCH_SIZE + 1
    mock_embed.return_value = [[0.1] * 1536] * BATCH_SIZE
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(n)
    for c in chunks:
        db.insert_chunk(c)

    # mock returns correct batch size each call
    def side_effect(texts):
        return [[0.1] * 1536] * len(texts)
    mock_embed.side_effect = side_effect

    embed_chunks(chunks, db)
    assert mock_embed.call_count == 2  # ceil(n / BATCH_SIZE)
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_embedder.py -v
```

Expected: `ModuleNotFoundError: No module named 'indexer.embedder'`

- [ ] **Step 3: Write `indexer/embedder.py`**

```python
from __future__ import annotations
import math
from openai import OpenAI
from storage.chunk import Chunk
from storage.db import DB

BATCH_SIZE = 100
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _openai_embed(texts: list[str]) -> list[list[float]]:
    response = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_chunks(chunks: list[Chunk], db: DB) -> None:
    """Embed chunks and write to DB. Skips chunks that already have an embedding."""
    # insert all chunks first, collect (chunk, rowid) pairs
    pairs: list[tuple[Chunk, int]] = []
    for chunk in chunks:
        rowid = db.insert_chunk(chunk)
        if not db.has_embedding(rowid):
            pairs.append((chunk, rowid))

    if not pairs:
        return

    # batch embed
    num_batches = math.ceil(len(pairs) / BATCH_SIZE)
    for i in range(num_batches):
        batch = pairs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        texts = [c.embed_text for c, _ in batch]
        embeddings = _openai_embed(texts)
        for (_, rowid), embedding in zip(batch, embeddings):
            db.insert_embedding(rowid, embedding)
        print(f"  embedded batch {i + 1}/{num_batches} ({len(batch)} chunks)")
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add indexer/embedder.py tests/test_embedder.py
git commit -m "feat: embedder — batched OpenAI embedding with idempotency gate"
```
