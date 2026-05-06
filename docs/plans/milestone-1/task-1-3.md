# Task 1.3: AST Chunker

**Files:**

- Create: `indexer/chunker.py`
- Create: `tests/test_chunker.py`

---

- [x] **Step 1: Write the failing tests**

`tests/test_chunker.py`:

```python
from pathlib import Path
import tempfile
from indexer.chunker import chunk_file, chunk_corpus

SAMPLE_SOURCE = '''\
"""Module docstring."""

def top_level_fn(x):
    """A top-level function."""
    return x + 1


class MyClass:
    """A class."""

    def method_a(self):
        """Method A."""
        pass

    def method_b(self, x, y):
        pass


async def async_fn():
    pass
'''


def _write_temp_py(source: str) -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp())
    f = tmp / "sample.py"
    f.write_text(source)
    return f, tmp


def test_chunk_file_top_level_functions():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    names = {c.symbol_name for c in chunks}
    assert "top_level_fn" in names
    assert "async_fn" in names


def test_chunk_file_class():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    class_chunks = [c for c in chunks if c.symbol_name == "MyClass"]
    assert len(class_chunks) == 1
    assert class_chunks[0].symbol_type == "class"


def test_chunk_file_methods_are_siblings():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    method_chunks = [c for c in chunks if c.symbol_type == "method"]
    names = {c.symbol_name for c in method_chunks}
    assert "method_a" in names
    assert "method_b" in names
    for mc in method_chunks:
        assert mc.parent_class == "MyClass"


def test_chunk_file_method_embed_text_prefixed():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    method_a = next(c for c in chunks if c.symbol_name == "method_a")
    assert method_a.embed_text.startswith("MyClass.method_a: ")


def test_chunk_file_docstring_extracted():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    fn = next(c for c in chunks if c.symbol_name == "top_level_fn")
    assert fn.docstring == "A top-level function."


def test_chunk_file_no_docstring_is_none():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    method_b = next(c for c in chunks if c.symbol_name == "method_b")
    assert method_b.docstring is None


def test_chunk_file_line_numbers_correct():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    fn = next(c for c in chunks if c.symbol_name == "top_level_fn")
    assert fn.line_start == 3
    assert fn.line_end == 5


def test_chunk_file_relative_path():
    f, root = _write_temp_py(SAMPLE_SOURCE)
    chunks = chunk_file(f, root)
    assert all(not c.file_path.startswith("/") for c in chunks)
    assert all(c.file_path.endswith(".py") for c in chunks)


def test_chunk_file_syntax_error_returns_empty():
    f, root = _write_temp_py("def broken(:\n    pass")
    chunks = chunk_file(f, root)
    assert chunks == []


def test_chunk_corpus_recurses():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.py").write_text("def fn_a(): pass")
    sub = tmp / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("def fn_b(): pass")
    chunks = chunk_corpus(tmp)
    names = {c.symbol_name for c in chunks}
    assert "fn_a" in names
    assert "fn_b" in names
```

- [x] **Step 2: Run tests — confirm failure**

```bash
pytest tests/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'indexer.chunker'`

- [x] **Step 3: Write `indexer/chunker.py`**

```python
from __future__ import annotations
import ast
from pathlib import Path
from storage.chunk import Chunk, make_chunk


def _get_docstring(node: ast.AST) -> str | None:
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _extract_source(lines: list[str], node: ast.AST) -> str:
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def chunk_file(file_path: Path, corpus_root: Path) -> list[Chunk]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    lines = source.splitlines()
    rel_path = str(file_path.relative_to(corpus_root)).replace("\\", "/")
    chunks: list[Chunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                make_chunk(
                    file_path=rel_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    parent_class=None,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=_get_docstring(node),
                    text=_extract_source(lines, node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            chunks.append(
                make_chunk(
                    file_path=rel_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    parent_class=None,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=_get_docstring(node),
                    text=_extract_source(lines, node),
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        make_chunk(
                            file_path=rel_path,
                            symbol_name=item.name,
                            symbol_type="method",
                            parent_class=node.name,
                            line_start=item.lineno,
                            line_end=item.end_lineno,
                            docstring=_get_docstring(item),
                            text=_extract_source(lines, item),
                        )
                    )

    return chunks


def chunk_corpus(corpus_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for py_file in sorted(corpus_root.rglob("*.py")):
        chunks.extend(chunk_file(py_file, corpus_root))
    return chunks
```

- [x] **Step 4: Run tests — confirm pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: 10 tests PASSED.

- [x] **Step 5: Commit**

```bash
git add indexer/chunker.py tests/test_chunker.py
git commit -m "feat: AST chunker — functions, classes, methods as sibling chunks"
```
