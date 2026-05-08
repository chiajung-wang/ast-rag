from ui.helpers import parse_citations, build_permalink


def test_parse_single_citation():
    text = "Defined in [runnables/base.py:10-50]."
    result = parse_citations(text)
    assert result == [("runnables/base.py", 10, 50)]


def test_parse_multiple_citations():
    text = "See [runnables/base.py:10-50] and [schema/runnable.py:1-20]."
    result = parse_citations(text)
    assert result == [("runnables/base.py", 10, 50), ("schema/runnable.py", 1, 20)]


def test_parse_no_citations():
    assert parse_citations("No citations here.") == []


def test_build_permalink():
    url = build_permalink("runnables/base.py", 10, 50)
    assert "langchain-ai/langchain" in url
    assert "runnables/base.py" in url
    assert "#L10-L50" in url
    assert "1519ed5afbc3bfcc7170b12baa07f1ae7e98edd0" in url
