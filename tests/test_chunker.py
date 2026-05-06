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
