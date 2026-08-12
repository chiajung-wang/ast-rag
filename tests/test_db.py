import struct
import tempfile
from pathlib import Path
import pytest
from storage.chunk import make_chunk
from storage.db import DB

EMBEDDING_DIM = 1536


def _fake_embedding(seed: float = 0.1) -> list[float]:
    return [seed] * EMBEDDING_DIM


def _serialize(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


@pytest.fixture
def db(tmp_path):
    return DB(str(tmp_path / "test.db"))


@pytest.fixture
def chunk_a():
    return make_chunk("runnables/base.py", "RunnableSequence", "class", None, 10, 100, "A runnable.", "class RunnableSequence:\n    pass")


@pytest.fixture
def chunk_b():
    return make_chunk("runnables/base.py", "invoke", "method", "RunnableSequence", 20, 40, None, "def invoke(self): pass")


def test_insert_chunk_returns_rowid(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    assert isinstance(rowid, int)
    assert rowid > 0


def test_insert_chunk_duplicate_returns_same_rowid(db, chunk_a):
    r1 = db.insert_chunk(chunk_a)
    r2 = db.insert_chunk(chunk_a)
    assert r1 == r2


def test_has_embedding_false_before_insert(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    assert db.has_embedding(rowid) is False


def test_has_embedding_true_after_insert(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    db.insert_embedding(rowid, _fake_embedding())
    assert db.has_embedding(rowid) is True


def test_vector_search_returns_chunks(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    db.insert_embedding(rowid, _fake_embedding(0.1))
    results = db.vector_search(_fake_embedding(0.1), k=5)
    assert len(results) == 1
    assert results[0].symbol_name == "RunnableSequence"


def test_symbol_lookup_exact(db, chunk_a):
    db.insert_chunk(chunk_a)
    result = db.symbol_lookup("RunnableSequence")
    assert result is not None
    assert result.symbol_name == "RunnableSequence"


def test_symbol_lookup_case_insensitive(db, chunk_a):
    db.insert_chunk(chunk_a)
    assert db.symbol_lookup("runnablesequence") is not None
    assert db.symbol_lookup("RUNNABLESEQUENCE") is not None


def test_symbol_lookup_miss_returns_none(db):
    assert db.symbol_lookup("DoesNotExist") is None


def test_symbol_lookup_prefers_class_over_method_and_function(db):
    """1296 names cover 2414 chunks — collisions need a stable tie-break."""
    db.insert_chunk(make_chunk("z/last.py", "Tee", "function", None, 1, 5, None, "def Tee(): pass"))
    db.insert_chunk(make_chunk("m/mid.py", "Tee", "method", "Runnable", 1, 5, None, "def Tee(self): pass"))
    db.insert_chunk(make_chunk("a/first.py", "Tee", "class", None, 1, 5, None, "class Tee: pass"))

    result = db.symbol_lookup("Tee")
    assert result.symbol_type == "class"
    assert result.file_path == "a/first.py"


def test_symbol_lookup_tie_break_orders_by_file_path(db):
    """Same symbol_type in two files — lowest file_path wins, deterministically."""
    db.insert_chunk(make_chunk("z/last.py", "ToolCall", "class", None, 1, 5, None, "class ToolCall: pass"))
    db.insert_chunk(make_chunk("a/first.py", "ToolCall", "class", None, 1, 5, None, "class ToolCall: ..."))

    assert db.symbol_lookup("ToolCall").file_path == "a/first.py"


def test_chunk_exists_at_hit(db, chunk_a):
    db.insert_chunk(chunk_a)
    # line_start=10, line_end=100 — query range entirely inside
    assert db.chunk_exists_at("runnables/base.py", 10, 100) is True


def test_chunk_exists_at_miss_wrong_file(db, chunk_a):
    db.insert_chunk(chunk_a)
    assert db.chunk_exists_at("other/file.py", 10, 100) is False


def test_all_chunks_returns_all(db, chunk_a, chunk_b):
    db.insert_chunk(chunk_a)
    db.insert_chunk(chunk_b)
    chunks = db.all_chunks()
    names = {c.symbol_name for c in chunks}
    assert "RunnableSequence" in names
    assert "invoke" in names


def test_all_symbol_names(db, chunk_a, chunk_b):
    db.insert_chunk(chunk_a)
    db.insert_chunk(chunk_b)
    names = db.all_symbol_names()
    assert "RunnableSequence" in names
    assert "invoke" in names


def test_schema_init_is_idempotent(tmp_path):
    path = str(tmp_path / "test.db")
    DB(path)  # first open
    DB(path)  # second open — must not raise


# ── inheritance-aware class outline (task 6.5) ───────────────────────────────

def _cls(db, name, file_path, bases=None, line=1):
    from storage.chunk import make_chunk
    db.insert_chunk(make_chunk(file_path, name, "class", None, line, line + 50,
                               None, f"class {name}: pass", base_classes=bases or []))


def _meth(db, name, parent, file_path, line=10):
    from storage.chunk import make_chunk
    db.insert_chunk(make_chunk(file_path, name, "method", parent, line, line + 2,
                               None, f"def {name}(self): pass"))


def test_class_outline_includes_inherited_methods(db):
    _cls(db, "Base", "m.py")
    _meth(db, "on_event", "Base", "m.py", 10)
    _cls(db, "Child", "m.py", bases=["Base"], line=100)
    _meth(db, "own", "Child", "m.py", 110)

    outline = db.class_outline("Child")
    names = [e.chunk.symbol_name for e in outline.entries]
    assert names == ["own", "on_event"]
    assert [e.inherited for e in outline.entries] == [False, True]
    assert outline.entries[1].defining_class == "Base"


def test_class_outline_walks_mixins(db):
    """The real case: events live on mixins, not on the class itself."""
    for m in ("LLMMixin", "ChainMixin"):
        _cls(db, m, "cb.py")
    _meth(db, "on_llm_start", "LLMMixin", "cb.py", 10)
    _meth(db, "on_chain_start", "ChainMixin", "cb.py", 20)
    _cls(db, "Handler", "cb.py", bases=["LLMMixin", "ChainMixin"], line=200)
    _meth(db, "ignore_llm", "Handler", "cb.py", 210)

    names = {e.chunk.symbol_name for e in db.class_outline("Handler").entries}
    assert names == {"ignore_llm", "on_llm_start", "on_chain_start"}


def test_class_outline_override_hides_base_method(db):
    _cls(db, "Base", "m.py")
    _meth(db, "run", "Base", "m.py", 10)
    _cls(db, "Child", "m.py", bases=["Base"], line=100)
    _meth(db, "run", "Child", "m.py", 110)

    entries = db.class_outline("Child").entries
    assert len(entries) == 1
    assert entries[0].defining_class == "Child"
    assert entries[0].inherited is False


def test_class_outline_does_not_merge_same_name_classes_in_other_files(db):
    """ToolCall, RunInfo, Tee, NoLock and ToolCallChunk each appear twice."""
    _cls(db, "ToolCall", "messages/tool.py")
    _meth(db, "from_tool", "ToolCall", "messages/tool.py", 10)
    _cls(db, "ToolCall", "messages/content.py")
    _meth(db, "from_content", "ToolCall", "messages/content.py", 10)

    outline = db.class_outline("ToolCall")
    names = [e.chunk.symbol_name for e in outline.entries]
    assert names == ["from_content"]  # content.py sorts first, and only its methods
    assert outline.class_chunk.file_path == "messages/content.py"


def test_class_outline_reports_external_bases(db):
    _cls(db, "Model", "m.py", bases=["BaseModel", "ABC"])
    outline = db.class_outline("Model")
    assert set(outline.external_bases) == {"BaseModel", "ABC"}


def test_class_outline_lists_direct_subclasses(db):
    _cls(db, "Base", "m.py")
    _cls(db, "AsyncImpl", "m.py", bases=["Base"], line=100)
    _cls(db, "SyncImpl", "m.py", bases=["Base"], line=200)
    _cls(db, "Unrelated", "m.py", bases=["Other"], line=300)

    subs = {c.symbol_name for c in db.class_outline("Base").subclasses}
    assert subs == {"AsyncImpl", "SyncImpl"}


def test_class_outline_stops_at_max_depth(db):
    _cls(db, "L0", "m.py"); _meth(db, "m0", "L0", "m.py", 10)
    _cls(db, "L1", "m.py", bases=["L0"], line=100); _meth(db, "m1", "L1", "m.py", 110)
    _cls(db, "L2", "m.py", bases=["L1"], line=200); _meth(db, "m2", "L2", "m.py", 210)
    _cls(db, "L3", "m.py", bases=["L2"], line=300); _meth(db, "m3", "L3", "m.py", 310)
    _cls(db, "L4", "m.py", bases=["L3"], line=400); _meth(db, "m4", "L4", "m.py", 410)

    names = {e.chunk.symbol_name for e in db.class_outline("L4", max_depth=2).entries}
    assert names == {"m4", "m3", "m2"}


def test_class_outline_survives_inheritance_cycle(db):
    _cls(db, "A", "m.py", bases=["B"]); _meth(db, "a", "A", "m.py", 10)
    _cls(db, "B", "m.py", bases=["A"], line=100); _meth(db, "b", "B", "m.py", 110)
    names = {e.chunk.symbol_name for e in db.class_outline("A").entries}
    assert names == {"a", "b"}


def test_class_outline_missing_class_is_falsy(db):
    outline = db.class_outline("DoesNotExist")
    assert not outline
    assert outline.class_chunk is None


def test_insert_chunk_refreshes_base_classes_without_new_rowid(db):
    """Re-indexing must fill the column in place, never orphan an embedding."""
    from storage.chunk import make_chunk
    args = ("m.py", "C", "class", None, 1, 5, None, "class C: pass")
    first = db.insert_chunk(make_chunk(*args))
    db.insert_embedding(first, [0.1] * 1536)

    second = db.insert_chunk(make_chunk(*args, base_classes=["Base"]))
    assert second == first
    assert db.has_embedding(first) is True
    assert db.symbol_lookup("C").base_classes == ["Base"]


def test_db_is_usable_from_a_second_thread(tmp_path):
    """Streamlit runs one script thread per browser session, and DB lives in a
    module-level global. With check_same_thread=True the second session raised
    ProgrammingError."""
    import threading
    path = str(tmp_path / "t.db")
    db = DB(path)
    db.insert_chunk(make_chunk("m.py", "C", "class", None, 1, 5, None, "class C: pass"))

    result: dict = {}

    def read():
        try:
            result["chunk"] = db.symbol_lookup("C")
        except Exception as exc:  # noqa: BLE001 - the failure mode under test
            result["error"] = exc

    t = threading.Thread(target=read)
    t.start()
    t.join()

    assert "error" not in result, result.get("error")
    assert result["chunk"].symbol_name == "C"


def test_db_concurrent_reads_from_many_threads(tmp_path):
    import threading
    db = DB(str(tmp_path / "t.db"))
    db.insert_chunk(make_chunk("m.py", "C", "class", None, 1, 5, None, "class C: pass"))

    errors: list = []

    def read():
        try:
            for _ in range(20):
                db.symbol_lookup("C")
                db.all_symbol_names()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors


def test_no_query_bypasses_the_lock():
    """Structural guard for the bug CI caught on 2026-08-12.

    check_same_thread=False only permits sharing a connection between
    threads; the lock is what makes it safe. Any statement issued outside
    _fetchall / _fetchone / _exec / _executescript / _commit reintroduces the
    InterfaceError seen on the Linux runner, where sqlite3.threadsafety is 1
    rather than the 3 found on macOS. Locally this passes either way, so a
    structural check is the only thing that catches it before CI does.
    """
    import ast
    import inspect
    import storage.db

    source = inspect.getsource(storage.db)
    tree = ast.parse(source)
    db_class = next(n for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "DB")

    allowed = {"__init__", "_fetchall", "_fetchone", "_exec", "_executescript", "_commit"}
    offenders = []
    for method in db_class.body:
        if not isinstance(method, ast.FunctionDef) or method.name in allowed:
            continue
        for node in ast.walk(method):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "conn"):
                offenders.append(f"{method.name} -> self.conn.{node.attr}")

    assert not offenders, "statements bypassing the lock: " + ", ".join(offenders)


# ── embedding-model guard ────────────────────────────────────────────────────

def test_embed_model_is_recorded_and_read_back(db):
    from storage.db import EMBED_MODEL_KEY
    db.set_meta(EMBED_MODEL_KEY, "openai/text-embedding-3-small")
    assert db.get_meta(EMBED_MODEL_KEY) == "openai/text-embedding-3-small"


def test_set_meta_overwrites(db):
    db.set_meta("k", "one")
    db.set_meta("k", "two")
    assert db.get_meta("k") == "two"


def test_assert_embed_model_passes_on_match(db):
    from storage.db import EMBED_MODEL_KEY
    db.set_meta(EMBED_MODEL_KEY, "openai/text-embedding-3-small")
    db.assert_embed_model("openai/text-embedding-3-small")  # must not raise


def test_assert_embed_model_raises_on_mismatch(db):
    """Two models do not share a vector space. Without this the query returns
    plausible, wrong neighbours and nothing complains."""
    from storage.db import EMBED_MODEL_KEY, EmbedModelMismatch
    db.set_meta(EMBED_MODEL_KEY, "openai/text-embedding-3-small")
    with pytest.raises(EmbedModelMismatch) as exc:
        db.assert_embed_model("openai/text-embedding-3-large")
    assert "text-embedding-3-small" in str(exc.value)
    assert "make index" in str(exc.value)


def test_assert_embed_model_allows_index_predating_the_guard(db):
    """An index built before the meta table existed records no model. Failing
    there would break every existing .db on upgrade."""
    db.assert_embed_model("openai/text-embedding-3-small")  # must not raise


def test_get_embedding_round_trips(db, chunk_a):
    """verify_embeddings.py compares stored vectors against fresh ones, so the
    read path must return what was written."""
    rowid = db.insert_chunk(chunk_a)
    vec = [i / 1000 for i in range(EMBEDDING_DIM)]
    db.insert_embedding(rowid, vec)

    got = db.get_embedding(rowid)
    assert got is not None and len(got) == EMBEDDING_DIM
    # stored as float32, so compare with tolerance rather than equality
    assert all(abs(a - b) < 1e-6 for a, b in zip(vec, got))


def test_get_embedding_missing_returns_none(db, chunk_a):
    rowid = db.insert_chunk(chunk_a)
    assert db.get_embedding(rowid) is None


def test_sample_embedded_chunks_skips_unembedded(db, chunk_a, chunk_b):
    embedded = db.insert_chunk(chunk_a)
    db.insert_embedding(embedded, _fake_embedding())
    db.insert_chunk(chunk_b)  # no embedding

    sample = db.sample_embedded_chunks(10)
    assert [c.symbol_name for _, c in sample] == ["RunnableSequence"]
    assert sample[0][0] == embedded


def test_sample_embedded_chunks_respects_limit(db):
    for i in range(5):
        c = make_chunk("m.py", f"C{i}", "class", None, i + 1, i + 2, None, f"class C{i}: pass")
        db.insert_embedding(db.insert_chunk(c), _fake_embedding())
    assert len(db.sample_embedded_chunks(3)) == 3
