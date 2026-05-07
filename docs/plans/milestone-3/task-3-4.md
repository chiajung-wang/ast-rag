# Task 3.4: `ask.py` CLI

**What to build:** CLI entry point that invokes the compiled LangGraph agent and prints the validated answer. No streaming — blocking `graph.invoke()`.

**Blocked by:** Task 3.3 (compiled graph)

**Acceptance criteria:**
- [ ] `python ask.py "Where is RunnableSequence defined?"` returns answer containing `[runnables/base.py:N-M]`
- [ ] `python ask.py "What is RunnablePassthrough?"` returns answer with ≥1 citation
- [ ] `python ask.py "foo bar baz nonsense"` returns graceful response, not an error
- [ ] No unit test — verified manually per acceptance criteria

---

**Files:**
- Create: `ask.py`

---

- [ ] **Step 1: Write `ask.py`**

```python
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
```

- [ ] **Step 2: Run acceptance checks**

```bash
python ask.py "Where is RunnableSequence defined?"
```
Expected: answer contains `[runnables/base.py:N-M]` citation.

```bash
python ask.py "What is RunnablePassthrough?"
```
Expected: answer with ≥1 citation.

```bash
python ask.py "foo bar baz nonsense"
```
Expected: graceful response like "I don't have source for this", no exception.

- [ ] **Step 3: Commit**

```bash
git add ask.py
git commit -m "feat(agent): ask.py CLI — blocking graph.invoke with printed answer"
```
