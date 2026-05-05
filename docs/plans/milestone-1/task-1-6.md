# Task 1.6: Clone Script

**What to build:** Script that clones `langchain-ai/langchain` at a pinned commit and extracts `libs/core/langchain_core/` into `CLONE_DIR`. Prints the commit SHA so the developer can update `corpus_config.py`.

**Blocked by:** Task 1.2 (corpus_config)

**Acceptance criteria:**
- [ ] `python -m indexer.clone` clones the repo if `CLONE_DIR` does not exist; skips if it does
- [ ] After running, `Path(CLONE_DIR) / CORPUS_SUBPATH` exists and contains `.py` files
- [ ] Prints the resolved commit SHA
- [ ] No unit test (git subprocess) — verified manually per acceptance criteria

---

**Files:**
- Create: `indexer/clone.py`

---

- [ ] **Step 1: Write `indexer/clone.py`**

```python
"""Clone langchain-core corpus.

Usage:
    python -m indexer.clone

After running, copy the printed SHA into indexer/corpus_config.py.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from indexer.corpus_config import REPO_URL, CLONE_DIR, CORPUS_SUBPATH, COMMIT_SHA


def clone_corpus() -> None:
    clone_path = Path(CLONE_DIR)

    if clone_path.exists():
        print(f"[clone] {CLONE_DIR} already exists — skipping clone.")
    else:
        print(f"[clone] Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--filter=blob:none", REPO_URL, CLONE_DIR],
            check=True,
        )

    # Checkout pinned commit if COMMIT_SHA is set
    if COMMIT_SHA and COMMIT_SHA != "FILL_IN_AFTER_CLONE":
        subprocess.run(
            ["git", "-C", CLONE_DIR, "checkout", COMMIT_SHA],
            check=True,
        )
        sha = COMMIT_SHA
    else:
        result = subprocess.run(
            ["git", "-C", CLONE_DIR, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        sha = result.stdout.strip()
        print(f"\n[clone] Resolved HEAD SHA: {sha}")
        print(f"[clone] Update COMMIT_SHA in indexer/corpus_config.py to pin this commit.\n")

    corpus_path = clone_path / CORPUS_SUBPATH
    if not corpus_path.exists():
        print(f"[clone] ERROR: {corpus_path} not found. Check CORPUS_SUBPATH.", file=sys.stderr)
        sys.exit(1)

    py_count = sum(1 for _ in corpus_path.rglob("*.py"))
    print(f"[clone] Corpus ready: {corpus_path} ({py_count} .py files)")


if __name__ == "__main__":
    clone_corpus()
```

- [ ] **Step 2: Run and verify manually**

```bash
python -m indexer.clone
```

Expected output (first run):
```
[clone] Cloning https://github.com/langchain-ai/langchain ...
[clone] Resolved HEAD SHA: <sha>
[clone] Update COMMIT_SHA in indexer/corpus_config.py to pin this commit.
[clone] Corpus ready: langchain/libs/core/langchain_core (N .py files)
```

- [ ] **Step 3: Update `indexer/corpus_config.py` with the printed SHA**

Edit `indexer/corpus_config.py`:
```python
COMMIT_SHA = "<paste SHA from output here>"
```

- [ ] **Step 4: Re-run — confirm skip behaviour**

```bash
python -m indexer.clone
```

Expected: `[clone] langchain already exists — skipping clone.`

- [ ] **Step 5: Commit**

```bash
git add indexer/clone.py indexer/corpus_config.py
git commit -m "feat: corpus clone script with pinned commit SHA"
```
