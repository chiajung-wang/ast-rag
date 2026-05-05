# Task 1.7: Indexer Pipeline + `query.py` CLI

**What to build:** `indexer/__main__.py` orchestrates clone → chunk → embed. `query.py` does a raw vector search against `index.db` for spot-checking.

**Blocked by:** Task 1.3 (chunker), Task 1.5 (embedder), Task 1.6 (clone)

**Acceptance criteria:**
- [ ] `make index` runs end-to-end without error
- [ ] `index.db` created and contains >1 000 chunks
- [ ] `python query.py "RunnableSequence"` returns ≥1 result with `symbol_name == "RunnableSequence"`
- [ ] `python query.py "invoke method"` returns ≥1 result with `symbol_type == "method"`
- [ ] Re-running `make index` is fast (skips already-embedded chunks)

---

**Files:**
- Create: `indexer/__main__.py`
- Create: `query.py`

---

- [ ] **Step 1: Write `indexer/__main__.py`**

```python
"""Indexer pipeline entry point.

Run via: python -m indexer  (or: make index)
"""
from __future__ import annotations
from pathlib import Path
from indexer.clone import clone_corpus
from indexer.chunker import chunk_corpus
from indexer.embedder import embed_chunks
from indexer.corpus_config import CLONE_DIR, CORPUS_SUBPATH, DB_PATH
from storage.db import DB


def build_index() -> None:
    # 1. ensure corpus is cloned
    clone_corpus()

    corpus_root = Path(CLONE_DIR) / CORPUS_SUBPATH
    print(f"\n[index] Chunking {corpus_root} ...")
    chunks = chunk_corpus(corpus_root)
    print(f"[index] {len(chunks)} chunks found.")

    # 2. embed + store
    db = DB(DB_PATH)
    print(f"[index] Embedding and storing (skipping existing) ...")
    embed_chunks(chunks, db)

    total = len(db.all_chunks())
    print(f"\n[index] Done. {total} chunks in {DB_PATH}.")


if __name__ == "__main__":
    build_index()
```

- [ ] **Step 2: Write `query.py`**

```python
#!/usr/bin/env python3
"""Raw retrieval spot-checker.

Usage:
    python query.py "Runnable definition"
    python query.py "invoke method" --k 10
"""
from __future__ import annotations
import sys
import argparse
from openai import OpenAI
from storage.db import DB
from indexer.corpus_config import DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    query = " ".join(args.query)

    client = OpenAI()
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding

    db = DB(DB_PATH)
    results = db.vector_search(embedding, k=args.k)

    print(f"\nQuery: {query!r}  ({len(results)} results)\n")
    for i, c in enumerate(results, 1):
        parent = f"{c.parent_class}." if c.parent_class else ""
        print(f"  {i}. [{c.symbol_type}] {parent}{c.symbol_name}")
        print(f"     {c.file_path}:{c.line_start}-{c.line_end}")
        if c.docstring:
            print(f"     {c.docstring[:80]}")
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the indexer**

```bash
make index
```

Expected output (first run, ~5–10 min for embedding):
```
[clone] langchain already exists — skipping clone.
[index] Chunking langchain/libs/core/langchain_core ...
[index] N chunks found.
[index] Embedding and storing (skipping existing) ...
  embedded batch 1/M (100 chunks)
  ...
[index] Done. N chunks in index.db.
```

- [ ] **Step 4: Verify chunk count**

```bash
python -c "from storage.db import DB; db = DB('index.db'); print(len(db.all_chunks()), 'chunks')"
```

Expected: `>1000 chunks`

- [ ] **Step 5: Spot-check with `query.py`**

```bash
python query.py "RunnableSequence"
python query.py "invoke method"
python query.py "how does streaming work"
```

For each: inspect that top results are relevant. At least one result per query should clearly match the intent.

- [ ] **Step 6: Verify idempotency**

```bash
make index  # second run — should be fast
```

Expected: `[index] Done. N chunks in index.db.` with no "embedded batch" lines (all skipped).

- [ ] **Step 7: Commit**

```bash
git add indexer/__main__.py query.py
git commit -m "feat: indexer pipeline — chunk + embed + store langchain-core"
```
