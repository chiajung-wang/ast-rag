from unittest.mock import patch, MagicMock
import json
from evals.run import compute_score, check_file_ok, format_results_md, run


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
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content=mock_answer)]}

    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content='{"path_correct": true, "path_reasoning": "ok", "description_correct": true, "missing_concepts": [], "forbidden_assertions_made": [], "description_reasoning": "ok", "overall_correct": true, "more_precise_than_reference": false}'
    )

    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatAnthropic", return_value=mock_judge_model):
        run(str(questions_path), str(results_path))

    content = results_path.read_text(encoding="utf-8")
    assert "q01" in content
    assert "Total:" in content


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

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"messages": [MagicMock(content="answer [runnables/base.py:1-5]")]}
    mock_judge_model = MagicMock()
    mock_judge_model.invoke.return_value = MagicMock(
        content='{"path_correct": true, "path_reasoning": "ok", "description_correct": true, "missing_concepts": [], "forbidden_assertions_made": [], "description_reasoning": "ok", "overall_correct": true, "more_precise_than_reference": false}'
    )

    with patch("evals.run.graph", mock_graph), \
         patch("evals.run.ChatAnthropic", return_value=mock_judge_model):
        run(str(questions_path), str(results_path))

    assert mock_graph.invoke.call_count == 1


def test_format_results_md_structure():
    rows = [
        {"id": "q01", "question": "Where is X?", "score": 2, "file_ok": True, "judge": "pass", "tier": "recall", "answer": "X is in [base.py:1-10]."},
        {"id": "q02", "question": "How does Y work?", "score": 0, "file_ok": False, "judge": "fail", "tier": "hard", "answer": ""},
    ]
    md = format_results_md(rows)
    assert "| id |" in md
    assert "| q01 |" in md
    assert "| q02 |" in md
    assert "Total: 2 / 4" in md
    assert "✓" in md
    assert "recall" in md
    assert "### q01" in md
    assert "X is in [base.py:1-10]." in md
