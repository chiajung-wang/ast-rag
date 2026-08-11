import json
import math

from storage.chunk import make_chunk
from evals.retrieval_eval import (
    load_questions,
    is_hit,
    hits,
    recall_at_k,
    mrr,
    ndcg_at_k,
    run_config,
    format_ablation_md,
    names_own_symbol,
    split_by_phrasing,
)


def _chunk(symbol_name: str, file_path: str = "runnables/base.py"):
    return make_chunk(file_path, symbol_name, "class", None, 1, 5, None, f"class {symbol_name}: pass")


# ── hit matching ──────────────────────────────────────────────────────────────

def test_is_hit_symbol_and_path_match():
    assert is_hit(_chunk("RunnableSequence"), ["RunnableSequence"], ["runnables/base.py"])


def test_is_hit_rejects_right_symbol_wrong_file():
    """Symbol names collide (invoke names 23 chunks) — the file must match too."""
    c = _chunk("invoke", "language_models/chat_models.py")
    assert not is_hit(c, ["invoke"], ["runnables/base.py"])


def test_is_hit_rejects_wrong_symbol():
    assert not is_hit(_chunk("RunnableLambda"), ["RunnableSequence"], ["runnables/base.py"])


def test_is_hit_any_expected_symbol_counts():
    """Gold is disjunctive: several symbols can answer one question."""
    c = _chunk("RunnableAssign", "runnables/passthrough.py")
    assert is_hit(c, ["RunnablePassthrough", "RunnableAssign"], ["runnables/passthrough.py"])


def test_hits_maps_over_ranked_list():
    chunks = [_chunk("A"), _chunk("RunnableSequence"), _chunk("B")]
    assert hits(chunks, ["RunnableSequence"], ["runnables/base.py"]) == [False, True, False]


# ── recall@k ──────────────────────────────────────────────────────────────────

def test_recall_at_k_all_hit():
    assert recall_at_k([[True, False], [False, True]], k=5) == 1.0


def test_recall_at_k_none_hit():
    assert recall_at_k([[False, False], [False, False]], k=5) == 0.0


def test_recall_at_k_half():
    assert recall_at_k([[True], [False]], k=5) == 0.5


def test_recall_at_k_respects_cutoff():
    """A hit at rank 6 counts for recall@10 but not for recall@5."""
    h = [[False] * 5 + [True]]
    assert recall_at_k(h, k=5) == 0.0
    assert recall_at_k(h, k=10) == 1.0


def test_recall_empty_question_set():
    assert recall_at_k([], k=5) == 0.0


# ── MRR ───────────────────────────────────────────────────────────────────────

def test_mrr_rank_one():
    assert mrr([[True, False, False]]) == 1.0


def test_mrr_rank_three():
    assert abs(mrr([[False, False, True]]) - 1 / 3) < 1e-9


def test_mrr_counts_only_first_hit():
    assert abs(mrr([[False, True, True]]) - 0.5) < 1e-9


def test_mrr_no_hit_scores_zero():
    assert mrr([[False, False]]) == 0.0


def test_mrr_averages_across_questions():
    # 1.0 and 0.5 -> 0.75
    assert abs(mrr([[True], [False, True]]) - 0.75) < 1e-9


# ── nDCG@k ────────────────────────────────────────────────────────────────────

def test_ndcg_perfect_when_hit_is_first():
    assert abs(ndcg_at_k([[True, False, False]], k=5) - 1.0) < 1e-9


def test_ndcg_discounts_lower_rank():
    # single hit at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1
    expected = (1 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k([[False, True]], k=5) - expected) < 1e-9


def test_ndcg_two_hits_at_top_is_perfect():
    assert abs(ndcg_at_k([[True, True, False]], k=5) - 1.0) < 1e-9


def test_ndcg_never_exceeds_one():
    """The ideal is built from hits actually found, so nDCG stays bounded."""
    for h in ([[True, True, True]], [[False, True, True]], [[True, False, True]]):
        assert ndcg_at_k(h, k=5) <= 1.0 + 1e-9


def test_ndcg_no_hit_scores_zero():
    assert ndcg_at_k([[False, False]], k=5) == 0.0


# ── loading ───────────────────────────────────────────────────────────────────

def test_load_questions_skips_meta_and_negative(tmp_path):
    lines = [
        {"_meta": "header"},
        {"id": "q01", "question": "Where is X?", "expected_file_paths": ["a.py"], "expected_symbols": ["X"]},
        {"id": "q34", "question": "Where is ChatOpenAI?", "expected_file_paths": [], "expected_symbols": []},
    ]
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    loaded = load_questions(str(p))
    assert [q["id"] for q in loaded] == ["q01"]


def test_all_shipped_questions_carry_gold_symbols():
    """Regression guard: a new question without labels silently leaves the eval."""
    graded = load_questions("evals/questions.jsonl")
    assert len(graded) == 33  # 34 total, negative tier excluded
    for q in graded:
        assert q["expected_symbols"], q["id"]


# ── run_config / reporting ────────────────────────────────────────────────────

def test_run_config_scores_a_perfect_retriever():
    questions = [
        {"id": "q01", "question": "Where is RunnableSequence defined?",
         "expected_symbols": ["RunnableSequence"], "expected_file_paths": ["runnables/base.py"]},
    ]
    perfect = lambda _q: [_chunk("RunnableSequence")]
    row = run_config("perfect", perfect, questions)
    assert row["recall@5"] == 1.0
    assert row["mrr"] == 1.0
    assert row["misses"] == []


def test_run_config_records_misses():
    questions = [
        {"id": "q01", "question": "Where is X?",
         "expected_symbols": ["X"], "expected_file_paths": ["a.py"]},
    ]
    useless = lambda _q: [_chunk("Unrelated")]
    row = run_config("useless", useless, questions)
    assert row["recall@5"] == 0.0
    assert row["misses"] == ["q01"]


def test_names_own_symbol_detects_verbatim_mention():
    q = {"question": "Where is RunnableSequence defined?", "expected_symbols": ["RunnableSequence"]}
    assert names_own_symbol(q)


def test_names_own_symbol_is_case_insensitive():
    q = {"question": "where is runnablesequence defined?", "expected_symbols": ["RunnableSequence"]}
    assert names_own_symbol(q)


def test_names_own_symbol_false_when_absent():
    q = {"question": "How do you chain two Runnables with the pipe operator?",
         "expected_symbols": ["__or__"]}
    assert not names_own_symbol(q)


def test_split_by_phrasing_separates_the_confound():
    questions = [
        {"id": "a", "question": "Where is Foo?", "expected_symbols": ["Foo"]},
        {"id": "b", "question": "Where is Bar?", "expected_symbols": ["Bar"]},
        {"id": "c", "question": "How does chaining work?", "expected_symbols": ["__or__"]},
    ]
    rows = [{"name": "cfg", "misses": ["c"]}]
    split = split_by_phrasing(rows, questions)[0]
    assert (split["naming_hit"], split["naming_total"]) == (2, 2)
    assert (split["other_hit"], split["other_total"]) == (0, 1)


def test_format_ablation_md_reports_the_confound():
    rows = [{"name": "cfg", "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "ndcg@5": 1.0, "misses": []}]
    questions = [
        {"id": "a", "question": "Where is Foo?", "expected_symbols": ["Foo"]},
        {"id": "c", "question": "How does chaining work?", "expected_symbols": ["__or__"]},
    ]
    md = format_ablation_md(rows, 2, questions)
    assert "Confound" in md
    assert "1 of 2 questions" in md


def test_format_ablation_md_has_all_rows():
    rows = [
        {"name": "BM25 only", "recall@5": 0.5, "recall@10": 0.6, "mrr": 0.4, "ndcg@5": 0.45, "misses": ["q02"]},
        {"name": "RRF hybrid", "recall@5": 0.8, "recall@10": 0.9, "mrr": 0.7, "ndcg@5": 0.75, "misses": []},
    ]
    md = format_ablation_md(rows, n_questions=33)
    assert "| BM25 only |" in md
    assert "| RRF hybrid |" in md
    assert "50.0%" in md
    assert "33 questions" in md
    assert "q02" in md
    assert "none" in md
