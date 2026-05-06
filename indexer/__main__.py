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
