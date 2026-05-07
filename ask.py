#!/usr/bin/env python3
"""Agent CLI.

Usage:
    python ask.py "Where is Runnable defined?"
"""
from __future__ import annotations
import argparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import graph

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+")
    args = parser.parse_args()
    question = " ".join(args.question)

    result = graph.invoke({
        "messages": [HumanMessage(content=question)],
        "retrieved_chunks": [],
    })

    answer = result["messages"][-1].content
    print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
