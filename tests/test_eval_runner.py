from unittest.mock import patch, MagicMock
import json
from evals.run import compute_score, check_file_ok, format_results_md, compute_cost, run


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
         patch("evals.run.ChatAnthropic", return_value=mock_judge_model):
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
         patch("evals.run.ChatAnthropic", return_value=mock_judge_model):
        run(str(questions_path), str(results_path), n_runs=1)

    assert mock_graph.invoke.call_count == 1


def test_format_results_md_structure():
    run1 = {"score": 2, "file_ok": True, "judge": "pass", "agent_cost": 0.0012, "judge_cost": 0.0003, "answer": "X is in [base.py:1-10].", "tool_trace": []}
    run2 = {"score": 0, "file_ok": False, "judge": "fail", "agent_cost": 0.0, "judge_cost": 0.0, "answer": "", "tool_trace": []}
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
    cost = compute_cost("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert abs(cost - 4.80) < 0.01


def test_compute_cost_unknown_model_fallback():
    cost = compute_cost("claude-unknown-99", input_tokens=1_000_000, output_tokens=0)
    assert abs(cost - 3.00) < 0.01
