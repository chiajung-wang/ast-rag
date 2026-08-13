import pytest
from storage.chunk import make_chunk
from storage.db import DB
from agent.citations import validate_citations, normalize_path


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


# ── path normalization ────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", [
    "runnables/base.py",
    "langchain_core/runnables/base.py",
    "core/langchain_core/runnables/base.py",
    "libs/core/langchain_core/runnables/base.py",
    "langchain/libs/core/langchain_core/runnables/base.py",
    "/langchain_core/runnables/base.py",
    "./langchain_core/runnables/base.py",
])
def test_normalize_path_strips_corpus_prefixes(path):
    assert normalize_path(path) == "runnables/base.py"


def test_normalize_path_leaves_unknown_prefix_alone():
    assert normalize_path("other/pkg/base.py") == "other/pkg/base.py"


def test_prefixed_citation_is_kept_not_stripped(db):
    """A correct citation written with the package prefix used to be removed."""
    text = "Defined in [langchain_core/runnables/base.py:10-100]."
    result = validate_citations(text, db)
    assert "could not be verified" not in result


def test_prefixed_citation_is_rewritten_to_canonical_path(db):
    """Downstream joins file_path onto the corpus root, so one form must survive."""
    text = "Defined in [langchain_core/runnables/base.py:10-100]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:10-100]" in result
    assert "langchain_core/runnables/base.py" not in result


def test_prefixed_citation_with_bad_range_still_stripped(db):
    text = "See [langchain_core/runnables/base.py:999-1000]."
    result = validate_citations(text, db)
    assert "999-1000" not in result
    assert "*1 citation(s) could not be verified and were removed.*" in result


# ── citation stats: precision, strip rate, hallucinated-path rate (6.7) ─────

from agent.citations import validate_citations_with_stats, CitationStats


def test_stats_counts_precise_citation(db):
    """The chunk overlapping the marker's range is named in the answer text
    -- precision should count it."""
    text = "RunnableSequence is defined in [runnables/base.py:10-100]."
    result, stats = validate_citations_with_stats(text, db)
    assert stats.emitted == 1
    assert stats.survived == 1
    assert stats.precise == 1
    assert stats.precision == 1.0
    assert stats.strip_rate == 0.0


def test_stats_imprecise_when_symbol_not_named_in_answer(db):
    """The marker's range is real (containment holds, so it survives), but
    the answer never names the symbol at that location -- e.g. a citation
    copy-pasted onto the wrong claim. Precision must catch what plain
    chunk_exists_at containment cannot."""
    text = "Something unrelated is described here [runnables/base.py:10-100]."
    result, stats = validate_citations_with_stats(text, db)
    assert stats.survived == 1
    assert stats.precise == 0
    assert stats.precision == 0.0


def test_stats_strip_rate_and_hallucinated_path_rate(db):
    text = "See [runnables/base.py:10-100] and [made/up/path.py:1-5]."
    result, stats = validate_citations_with_stats(text, db)
    assert stats.emitted == 2
    assert stats.stripped == 1
    assert stats.strip_rate == 0.5
    assert stats.hallucinated_paths == 1
    assert stats.hallucinated_path_rate == 0.5


def test_stats_wrong_range_on_real_path_is_not_hallucinated(db):
    """A bad line range on a real, indexed file is a different failure mode
    from inventing a file that was never indexed -- only the latter counts
    as a hallucinated path."""
    text = "See [runnables/base.py:9000-9100]."
    result, stats = validate_citations_with_stats(text, db)
    assert stats.stripped == 1
    assert stats.hallucinated_paths == 0


def test_stats_no_citations_returns_zeroed_stats(db):
    result, stats = validate_citations_with_stats("No citations here.", db)
    assert stats == CitationStats(emitted=0, stripped=0, hallucinated_paths=0, precise=0, survived=0)
    assert stats.precision == 0.0
    assert stats.strip_rate == 0.0


def test_validate_citations_unchanged_by_the_stats_refactor(db):
    """validate_citations is now a thin wrapper over the stats function --
    its own text-only behavior must be identical to before."""
    text = "See [runnables/base.py:999-1000]."
    assert "999-1000" not in validate_citations(text, db)
