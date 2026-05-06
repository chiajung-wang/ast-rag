# Task 1.2: Chunk Dataclass + Corpus Config

**Files:**

- Create: `storage/chunk.py`
- Create: `indexer/corpus_config.py`
- Create: `tests/test_chunk.py`

---

- [x] **Step 1: Write the failing test**

`tests/test_chunk.py`:

```python
from storage.chunk import Chunk, make_chunk

def test_make_chunk_function():
    c = make_chunk(
        file_path="runnables/base.py",
        symbol_name="invoke",
        symbol_type="function",
        parent_class=None,
        line_start=10,
        line_end=20,
        docstring="Invoke the runnable.",
        text="def invoke(self):\n    pass",
    )
    assert c.file_path == "runnables/base.py"
    assert c.symbol_name == "invoke"
    assert c.symbol_type == "function"
    assert c.parent_class is None
    assert c.line_start == 10
    assert c.line_end == 20
    assert c.docstring == "Invoke the runnable."
    assert c.embed_text == "def invoke(self):\n    pass"  # no prefix for top-level
    assert len(c.id) == 64  # sha256 hex


def test_make_chunk_method_prefixes_embed_text():
    c = make_chunk(
        file_path="runnables/base.py",
        symbol_name="invoke",
        symbol_type="method",
        parent_class="RunnableSequence",
        line_start=100,
        line_end=120,
        docstring=None,
        text="def invoke(self, input):\n    pass",
    )
    assert c.parent_class == "RunnableSequence"
    assert c.embed_text.startswith("RunnableSequence.invoke: ")
    assert "def invoke" in c.embed_text


def test_make_chunk_id_is_deterministic():
    kwargs = dict(
        file_path="a.py", symbol_name="foo", symbol_type="function",
        parent_class=None, line_start=1, line_end=5, docstring=None,
        text="def foo(): pass",
    )
    c1 = make_chunk(**kwargs)
    c2 = make_chunk(**kwargs)
    assert c1.id == c2.id


def test_make_chunk_different_content_different_id():
    c1 = make_chunk("a.py", "foo", "function", None, 1, 5, None, "def foo(): pass")
    c2 = make_chunk("a.py", "foo", "function", None, 1, 5, None, "def foo(): return 1")
    assert c1.id != c2.id
```

- [x] **Step 2: Run test — confirm failure**

```bash
pytest tests/test_chunk.py -v
```

Expected: `ModuleNotFoundError: No module named 'storage.chunk'`

- [x] **Step 3: Write `storage/chunk.py`**

```python
from __future__ import annotations
import hashlib
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    file_path: str        # relative to corpus root: "runnables/base.py"
    symbol_name: str      # "invoke"
    symbol_type: str      # "function" | "class" | "method"
    parent_class: str | None  # "RunnableSequence" if method, else None
    line_start: int
    line_end: int
    docstring: str | None
    text: str             # raw source of the symbol
    embed_text: str       # text used for BM25 + dense embedding


def make_chunk(
    file_path: str,
    symbol_name: str,
    symbol_type: str,
    parent_class: str | None,
    line_start: int,
    line_end: int,
    docstring: str | None,
    text: str,
) -> Chunk:
    embed_text = (
        f"{parent_class}.{symbol_name}: {text}" if parent_class else text
    )
    chunk_id = hashlib.sha256(
        f"{file_path}{symbol_name}{line_start}{line_end}{text}".encode()
    ).hexdigest()
    return Chunk(
        id=chunk_id,
        file_path=file_path,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        parent_class=parent_class,
        line_start=line_start,
        line_end=line_end,
        docstring=docstring,
        text=text,
        embed_text=embed_text,
    )
```

- [x] **Step 4: Write `indexer/corpus_config.py`**

```python
# Fill COMMIT_SHA after cloning: git -C langchain rev-parse HEAD
COMMIT_SHA = "FILL_IN_AFTER_CLONE"
REPO_URL = "https://github.com/langchain-ai/langchain"
CORPUS_SUBPATH = "libs/core/langchain_core"  # inside the cloned repo
CLONE_DIR = "langchain"                       # local clone target
DB_PATH = "index.db"
```

- [x] **Step 5: Run test — confirm pass**

```bash
pytest tests/test_chunk.py -v
```

Expected: 4 tests PASSED.

- [x] **Step 6: Commit**

```bash
git add storage/chunk.py indexer/corpus_config.py tests/test_chunk.py
git commit -m "feat: Chunk dataclass with sha256 id and method embed_text prefix"
```
