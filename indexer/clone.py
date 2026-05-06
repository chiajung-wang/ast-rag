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
