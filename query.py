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
