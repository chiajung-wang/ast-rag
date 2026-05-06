# Task 2.4: Update `query.py` + Spot Checks

**What to build:** Update `query.py` to call `search_corpus` from `retrieval/pipeline.py` (full hybrid BM25 + dense + RRF) instead of `db.vector_search` directly. Verify milestone 2 completion criteria manually.

**Blocked by:** Task 2.3 (retrieval pipeline)

**Acceptance criteria:**
- [ ] `python query.py "RunnableSequence"` — top result has `symbol_name == "RunnableSequence"`
- [ ] `python query.py "how does streaming work"` — ≥3 results with `symbol_type` of `"method"` or `"function"`
- [ ] `python query.py "invoke method" --k 10` — ≥1 result with `symbol_type == "method"`
- [ ] No unit test — verified manually per acceptance criteria

---

**Files:**
- Modify: `query.py`

---

- [ ] **Step 1: Update `query.py`**

Replace the current direct DB + `vector_search` implementation with `search_corpus`:

```python
#!/usr/bin/env python3
"""Raw retrieval spot-checker.

Usage:
    python query.py "Runnable definition"
    python query.py "invoke method" --k 10
"""
from __future__ import annotations
import argparse
from dotenv import load_dotenv
from retrieval.pipeline import search_corpus

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    query = " ".join(args.query)

    results = search_corpus(query, k=args.k)

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

- [ ] **Step 2: Run spot checks**

```bash
python query.py "RunnableSequence"
```
Expected: top result is `[class] RunnableSequence` or a `RunnableSequence` method.

```bash
python query.py "how does streaming work"
```
Expected: ≥3 results with `symbol_type` of `method` or `function`.

```bash
python query.py "invoke method" --k 10
```
Expected: ≥1 result with `[method]` label.

- [ ] **Step 3: Commit**

```bash
git add query.py
git commit -m "feat(retrieval): update query.py to use hybrid search_corpus"
```
