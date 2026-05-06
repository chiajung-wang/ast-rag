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
