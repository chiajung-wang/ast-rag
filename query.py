#!/usr/bin/env python3
"""Raw retrieval spot-checker.

Usage:
    python query.py "Runnable definition"
    python query.py "invoke method" --k 10
"""
from __future__ import annotations
import sys
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from storage.db import DB
from indexer.corpus_config import DB_PATH

load_dotenv()


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
