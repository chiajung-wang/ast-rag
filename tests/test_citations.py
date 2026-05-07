import pytest
from storage.chunk import make_chunk
from storage.db import DB
from agent.citations import validate_citations


@pytest.fixture
def db(tmp_path):
    d = DB(str(tmp_path / "test.db"))
    chunk = make_chunk("runnables/base.py", "RunnableSequence", "class", None, 10, 100, None, "class RS: pass")
    d.insert_chunk(chunk)
    return d


def test_valid_citation_unchanged(db):
    text = "Defined in [runnables/base.py:10-100]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:10-100]" in result
    assert "could not be verified" not in result


def test_invalid_citation_stripped(db):
    text = "See [runnables/base.py:999-1000] for details."
    result = validate_citations(text, db)
    assert "[runnables/base.py:999-1000]" not in result
    assert "*1 citation(s) could not be verified and were removed.*" in result


def test_multiple_invalid_citations_stripped(db):
    text = "See [runnables/base.py:999-1000] and [other/file.py:1-5]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:999-1000]" not in result
    assert "[other/file.py:1-5]" not in result
    assert "*2 citation(s) could not be verified and were removed.*" in result


def test_no_citations_unchanged(db):
    text = "This answer has no citations."
    result = validate_citations(text, db)
    assert result == text


def test_mixed_valid_and_invalid(db):
    text = "Valid [runnables/base.py:10-100] and invalid [fake/path.py:1-2]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:10-100]" in result
    assert "[fake/path.py:1-2]" not in result
    assert "*1 citation(s) could not be verified and were removed.*" in result
