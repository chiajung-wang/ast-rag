from unittest.mock import patch, MagicMock
import json
import pytest
from evals.run import compute_score, check_file_ok, format_results_md, compute_cost, run, _run_once


def test_score_both_pass():
    assert compute_score(file_ok=True, judge_pass=True) == 2


def test_score_file_only():
    assert compute_score(file_ok=True, judge_pass=False) == 1


def test_score_judge_only():
    assert compute_score(file_ok=False, judge_pass=True) == 1


def test_score_neither():
    assert compute_score(file_ok=False, judge_pass=False) == 0


def test_check_file_ok_first_path_matches():
    answer = "See [runnables/base.py:10-50] for details."
    assert check_file_ok(["runnables/base.py", "other.py"], answer) is True


def test_check_file_ok_second_path_matches():
    answer = "See [runnables/base.py:10-50] for details."
    assert check_file_ok(["missing.py", "runnables/base.py"], answer) is True


def test_check_file_ok_none_match():
    answer = "See [other/file.py:1-5] for details."
    assert check_file_ok(["runnables/base.py"], answer) is False


def test_run_writes_results_md(tmp_path):
    questions = [
        {
            "id": "q01",
            "question": "Where is RunnableSequence defined?",
            "expected_file_paths": ["runnables/base.py"],
            "description_must_include": ["composition primitive for chaining runnables"],
            "description_must_not_assert": [],
            "tier": "recall",
            "subsystem": "runnables",
        }
    ]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps(questions[0]) + "\n")
    results_path = tmp_path / "results.md"

    mock_answer = "RunnableSequence is in [runnables/base.py:10-50]."
    mock_last_msg = MagicMock()
    mock_last_msg.content = mock_answer
    mock_last_msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    mock_last_msg.additional_kwargs = {"tool_trace": [], "budget_exhausted": False}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [mock_last_msg]}

    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content='{"path_correct": true, "path_reasoning": "ok", "description_correct": true, "missing_concepts": [], "forbidden_assertions_made": [], "description_reasoning": "ok", "overall_correct": true, "more_precise_than_reference": false}',
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
    )

    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        run(str(questions_path), str(results_path), n_runs=1)

    content = results_path.read_text(encoding="utf-8")
    assert "q01" in content
    assert "Median total:" in content
    assert "Agent:" in content
    assert "$" in content


def test_run_skips_meta_lines(tmp_path):
    lines = [
        json.dumps({"_meta": "header — skip me"}),
        json.dumps({
            "id": "q01",
            "question": "Where is X?",
            "expected_file_paths": ["runnables/base.py"],
            "description_must_include": ["something"],
            "description_must_not_assert": [],
            "tier": "recall",
            "subsystem": "runnables",
        }),
    ]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("\n".join(lines) + "\n")
    results_path = tmp_path / "results.md"

    mock_last_msg = MagicMock()
    mock_last_msg.content = "answer [runnables/base.py:1-5]"
    mock_last_msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
    mock_last_msg.additional_kwargs = {"tool_trace": [], "budget_exhausted": False}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [mock_last_msg]}
    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content='{"path_correct": true, "path_reasoning": "ok", "description_correct": true, "missing_concepts": [], "forbidden_assertions_made": [], "description_reasoning": "ok", "overall_correct": true, "more_precise_than_reference": false}',
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
    )

    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        run(str(questions_path), str(results_path), n_runs=1)

    assert mock_graph.invoke.call_count == 1


def test_format_results_md_structure():
    cs_empty = {"emitted": 0, "stripped": 0, "hallucinated_paths": 0, "precise": 0, "survived": 0}
    run1 = {"score": 2, "file_ok": True, "judge": "pass", "agent_cost": 0.0012, "judge_cost": 0.0003,
            "answer": "X is in [base.py:1-10].", "tool_trace": [],
            "latency_s": 1.2, "rounds": 1, "budget_exhausted": False, "citation_stats": cs_empty}
    run2 = {"score": 0, "file_ok": False, "judge": "fail", "agent_cost": 0.0, "judge_cost": 0.0,
            "answer": "", "tool_trace": [],
            "latency_s": 0.5, "rounds": 0, "budget_exhausted": False, "citation_stats": cs_empty}
    rows = [
        {"id": "q01", "question": "Where is X?", "tier": "recall", "median_score": 2.0, "variance": 0.0, "agent_cost": 0.0012, "judge_cost": 0.0003, "runs": [run1]},
        {"id": "q02", "question": "How does Y work?", "tier": "hard", "median_score": 0.0, "variance": 0.0, "agent_cost": 0.0, "judge_cost": 0.0, "runs": [run2]},
    ]
    md = format_results_md(rows, n_runs=1)
    assert "| id |" in md
    assert "| q01 |" in md
    assert "| q02 |" in md
    assert "Median total:" in md
    assert "Agent:" in md
    assert "recall" in md
    assert "### q01" in md
    assert "X is in [base.py:1-10]." in md


def test_compute_cost_haiku():
    # haiku-4.5 list price: $1.00 in / $5.00 out per MTok
    cost = compute_cost("anthropic/claude-haiku-4.5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 6.00) < 0.01


def test_compute_cost_opus():
    # opus-4.7 list price: $5.00 in / $25.00 out per MTok
    cost = compute_cost("anthropic/claude-opus-4.7", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 30.00) < 0.01


def test_run_once_agent_error_returns_zero_score_record():
    """A raising graph must not kill the run — tool_trace was unbound here."""
    question = {
        "id": "q01",
        "question": "Where is X?",
        "expected_file_paths": ["runnables/base.py"],
        "description_must_include": [],
        "description_must_not_assert": [],
        "tier": "recall",
    }
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("boom")

    with patch("evals.run.graph", mock_graph):
        result = _run_once(question)

    assert result["score"] == 0
    assert result["judge"] == "error"
    assert result["tool_trace"] == []
    assert result["answer"] == ""


def test_run_survives_agent_error(tmp_path):
    """A failing question must not stop the loop — results still get written."""
    question = {
        "id": "q01",
        "question": "Where is X?",
        "expected_file_paths": ["runnables/base.py"],
        "description_must_include": [],
        "description_must_not_assert": [],
        "tier": "recall",
        "subsystem": "runnables",
    }
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps(question) + "\n")
    results_path = tmp_path / "results.md"

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("boom")

    with patch("evals.run.graph", mock_graph):
        run(str(questions_path), str(results_path), n_runs=1)

    content = results_path.read_text(encoding="utf-8")
    assert "q01" in content
    assert "Median total: 0" in content


def test_compute_cost_unknown_model_fallback():
    cost = compute_cost("claude-unknown-99", input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 3.00) < 0.01


# ── latency/rounds/citation instrumentation (task 6.7) ───────────────────────

from evals.run import percentile, aggregate_by_tier, format_tier_report_md


def test_percentile_p50_is_median():
    assert percentile([1, 2, 3, 4, 5], 50) == 3


def test_percentile_p95_near_top_of_range():
    data = list(range(1, 101))  # 1..100
    assert percentile(data, 95) == pytest.approx(95.05, abs=0.5)


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_single_value():
    assert percentile([7.0], 95) == 7.0


def _run(latency_s=1.0, rounds=2, budget_exhausted=False, emitted=2, stripped=0,
         hallucinated=0, precise=2, survived=2):
    return {
        "latency_s": latency_s, "rounds": rounds, "budget_exhausted": budget_exhausted,
        "citation_stats": {
            "emitted": emitted, "stripped": stripped, "hallucinated_paths": hallucinated,
            "precise": precise, "survived": survived,
        },
    }


def test_aggregate_by_tier_groups_correctly():
    rows = [
        {"tier": "recall", "runs": [_run(latency_s=1.0), _run(latency_s=3.0)]},
        {"tier": "hard", "runs": [_run(latency_s=10.0)]},
    ]
    agg = {a["tier"]: a for a in aggregate_by_tier(rows)}
    assert agg["recall"]["n_runs"] == 2
    assert agg["hard"]["n_runs"] == 1
    assert agg["recall"]["p50_latency_s"] == 2.0


def test_aggregate_by_tier_exhausted_rate():
    rows = [{"tier": "hard", "runs": [
        _run(budget_exhausted=True), _run(budget_exhausted=False),
        _run(budget_exhausted=False), _run(budget_exhausted=False),
    ]}]
    agg = aggregate_by_tier(rows)[0]
    assert agg["exhausted_rate"] == 0.25


def test_aggregate_by_tier_citation_rates_are_pooled_not_averaged():
    """Pooled: sum(stripped)/sum(emitted) across runs, not mean of per-run
    rates -- a run with 0 citations must not contribute a spurious 0% or
    divide-by-zero to the average."""
    rows = [{"tier": "recall", "runs": [
        _run(emitted=4, stripped=1, precise=2, survived=3),
        _run(emitted=0, stripped=0, precise=0, survived=0),  # no citations this run
        _run(emitted=6, stripped=1, precise=4, survived=5),
    ]}]
    agg = aggregate_by_tier(rows)[0]
    # pooled: stripped=2/emitted=10 = 20%; precision = precise=6/survived=8 = 75%
    assert agg["strip_rate"] == pytest.approx(0.2)
    assert agg["precision"] == pytest.approx(0.75)


def test_aggregate_by_tier_handles_zero_citations_in_tier():
    rows = [{"tier": "recall", "runs": [_run(emitted=0, stripped=0, precise=0, survived=0)]}]
    agg = aggregate_by_tier(rows)[0]
    assert agg["strip_rate"] == 0.0
    assert agg["precision"] == 0.0


def test_format_tier_report_md_has_all_columns():
    rows = [{"tier": "recall", "runs": [_run()]}]
    md = format_tier_report_md(rows)
    assert "p50 latency" in md
    assert "p95 latency" in md
    assert "mean rounds" in md
    assert "exhausted%" in md
    assert "citation precision" in md
    assert "| recall |" in md


def test_format_tier_report_md_sorts_tiers():
    rows = [{"tier": "hard", "runs": [_run()]}, {"tier": "behavior", "runs": [_run()]}]
    md = format_tier_report_md(rows)
    assert md.index("| behavior |") < md.index("| hard |")


def test_format_results_md_includes_tier_report():
    """format_results_md's output must actually carry the tier report --
    wiring it in is easy to get half-done (function exists, never called)."""
    cs = {"emitted": 0, "stripped": 0, "hallucinated_paths": 0, "precise": 0, "survived": 0}
    run1 = {"score": 2, "file_ok": True, "judge": "pass", "agent_cost": 0.0, "judge_cost": 0.0,
            "answer": "x", "tool_trace": [], "latency_s": 1.0, "rounds": 1,
            "budget_exhausted": False, "citation_stats": cs}
    rows = [{"id": "q01", "question": "Q?", "tier": "recall", "median_score": 2.0,
             "variance": 0.0, "agent_cost": 0.0, "judge_cost": 0.0, "runs": [run1]}]
    md = format_results_md(rows, n_runs=1)
    assert "Per-tier instrumentation" in md
    assert "| recall |" in md


# ── regression gate (task 6.7) ────────────────────────────────────────────────

from evals.run import (
    total_score, load_baseline, save_baseline, baseline_key_for, check_regression, main,
)


def test_total_score_sums_median_and_caps_negative_at_one():
    rows = [
        {"median_score": 2.0, "tier": "recall"},
        {"median_score": 1.0, "tier": "negative"},
    ]
    total, max_total = total_score(rows)
    assert total == 3.0
    assert max_total == 3  # 2 (recall) + 1 (negative cap)


def test_baseline_key_for_dev_and_test_files():
    assert baseline_key_for("evals/questions.jsonl") == "dev"
    assert baseline_key_for("evals/questions-test.jsonl") == "test"


def test_baseline_key_for_unknown_file_is_its_own_name():
    assert baseline_key_for("evals/scratch.jsonl") == "scratch.jsonl"


def test_load_baseline_missing_file_is_empty_dict(tmp_path):
    assert load_baseline(tmp_path / "nope.json") == {}


def test_save_then_load_baseline_round_trips(tmp_path):
    p = tmp_path / "baseline.json"
    save_baseline({"dev": {"median_total": 90, "max_total": 95, "n_runs": 3}}, p)
    assert load_baseline(p)["dev"]["median_total"] == 90


def test_check_regression_passes_within_tolerance():
    baseline = {"dev": {"median_total": 91.0, "max_total": 95, "n_runs": 3}}
    passed, msg = check_regression(89.5, 95, "dev", baseline)  # 1.5 point drop
    assert passed is True


def test_check_regression_fails_beyond_tolerance():
    baseline = {"dev": {"median_total": 91.0, "max_total": 95, "n_runs": 3}}
    passed, msg = check_regression(85.0, 95, "dev", baseline)  # 6 point drop
    assert passed is False
    assert "REGRESSION" in msg


def test_check_regression_improvement_always_passes():
    baseline = {"dev": {"median_total": 91.0, "max_total": 95, "n_runs": 3}}
    passed, _ = check_regression(95.0, 95, "dev", baseline)
    assert passed is True


def test_check_regression_no_baseline_entry_passes_with_note():
    passed, msg = check_regression(50.0, 60, "test", {})
    assert passed is True
    assert "no baseline" in msg


def test_check_regression_mismatched_question_count_skips_gate():
    """Question set grew from 33 to 45 mid-project (task 6.4). A raw score
    comparison across different max_total values is meaningless -- the gate
    must skip, not silently compare apples to oranges, and it must not treat
    a legitimate question-set change as a regression."""
    baseline = {"dev": {"median_total": 91.0, "max_total": 95, "n_runs": 3}}
    passed, msg = check_regression(20.0, 33, "dev", baseline)
    assert passed is True
    assert "question set changed" in msg


def test_real_baseline_json_has_the_measured_dev_and_test_numbers():
    """Guards against the baseline ever silently reverting to the null
    placeholders task 6.7's own template shipped with."""
    baseline = load_baseline()
    assert baseline["dev"]["median_total"] is not None
    assert baseline["dev"]["max_total"] == 95
    assert baseline["test"]["median_total"] is not None


def test_run_returns_rows_and_respects_isolated_baseline_path(tmp_path):
    """run() must not touch the real repo evals/baseline.json when a test
    points it elsewhere."""
    questions = [{
        "id": "q01", "question": "Where is X?",
        "expected_file_paths": ["runnables/base.py"],
        "description_must_include": ["thing"], "description_must_not_assert": [],
        "tier": "recall", "subsystem": "runnables",
    }]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps(questions[0]) + "\n")
    results_path = tmp_path / "results.md"
    isolated_baseline = tmp_path / "baseline.json"
    save_baseline({"dev": {"median_total": 1.0, "max_total": 1, "n_runs": 1}}, isolated_baseline)

    mock_last_msg = MagicMock()
    mock_last_msg.content = "X is in [runnables/base.py:1-5]."
    mock_last_msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    mock_last_msg.additional_kwargs = {"tool_trace": [], "budget_exhausted": False}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [mock_last_msg]}
    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content='{"description_correct": true}',
        usage_metadata={"input_tokens": 10, "output_tokens": 5},
    )

    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        rows = run(str(questions_path), str(results_path), n_runs=1, baseline_path=isolated_baseline)

    assert len(rows) == 1
    assert rows[0]["id"] == "q01"


# ── main() exit code — the actual gate contract "make eval exits non-zero" ──

def _mock_full_run(score_json='{"description_correct": true}'):
    mock_last_msg = MagicMock()
    mock_last_msg.content = "X is in [runnables/base.py:1-5]."
    mock_last_msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    mock_last_msg.additional_kwargs = {"tool_trace": [], "budget_exhausted": False}
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [mock_last_msg]}
    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content=score_json, usage_metadata={"input_tokens": 10, "output_tokens": 5},
    )
    return mock_graph, mock_judge_model


def test_main_exits_zero_with_no_baseline_file(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps({
        "id": "q01", "question": "Where is X?", "expected_file_paths": ["runnables/base.py"],
        "description_must_include": ["thing"], "description_must_not_assert": [], "tier": "recall",
    }) + "\n")
    monkeypatch.chdir(tmp_path)  # so evals/baseline.json (default path) resolves to an empty tmp dir

    mock_graph, mock_judge_model = _mock_full_run()
    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        code = main(["--questions", str(questions_path), "--runs", "1",
                     "--results-dir", str(tmp_path / "results")])
    assert code == 0


def test_main_exits_one_on_a_real_regression(tmp_path, monkeypatch):
    """The actual contract task 6.7 asks for: make eval exits non-zero when
    the total score falls below a committed baseline. This drives it through
    main() exactly as the CLI does, not just check_regression() in isolation.

    2 questions, not 1: a 1-question set caps the max possible drop at 2.0
    points, and the gate requires a drop strictly greater than the 2.0
    tolerance -- a single failing question can never trigger it on its own.
    """
    questions = [
        {"id": "q01", "question": "Where is X?", "expected_file_paths": ["runnables/base.py"],
         "description_must_include": ["thing"], "description_must_not_assert": [], "tier": "recall"},
        {"id": "q02", "question": "Where is Y?", "expected_file_paths": ["runnables/base.py"],
         "description_must_include": ["thing"], "description_must_not_assert": [], "tier": "recall"},
    ]
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    # Baseline expects a perfect 4/4 across both questions; the answer below
    # fails file_ok (never mentions the expected path) and the judge below
    # fails description_correct, so both questions score 0/2 -- a 4-point
    # drop, well past the 2.0 tolerance.
    (tmp_path / "evals" / "baseline.json").write_text(
        json.dumps({"dev": {"median_total": 4.0, "max_total": 4, "n_runs": 1}}))

    mock_graph, mock_judge_model = _mock_full_run(score_json='{"description_correct": false}')
    mock_graph.invoke.return_value["messages"][0].content = "I don't have source for this."
    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        code = main(["--questions", str(questions_path), "--runs", "1",
                     "--results-dir", str(tmp_path / "results")])
    assert code == 1


def test_main_update_baseline_writes_and_exits_zero(tmp_path, monkeypatch):
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(json.dumps({
        "id": "q01", "question": "Where is X?", "expected_file_paths": ["runnables/base.py"],
        "description_must_include": ["thing"], "description_must_not_assert": [], "tier": "recall",
    }) + "\n")
    monkeypatch.chdir(tmp_path)

    mock_graph, mock_judge_model = _mock_full_run()
    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatOpenAI", return_value=mock_judge_model):
        code = main(["--questions", str(questions_path), "--runs", "1",
                     "--results-dir", str(tmp_path / "results"), "--update-baseline"])
    assert code == 0
    written = load_baseline(tmp_path / "evals" / "baseline.json")
    assert written["dev"]["max_total"] == 2  # 1 non-negative question -> max 2
