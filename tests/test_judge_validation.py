import json

from evals.judge_validation import (
    parse_results_md,
    load_question_rubrics,
    stratified_sample,
    write_blind_file,
    write_key_file,
    cohens_kappa,
)

_SAMPLE_MD = """\
| id | question | median | var | file_ok% | judge% | tier | agent_cost | judge_cost |
|---|---|---|---|---|---|---|---|---|
| q01 | Where is X? | 2 | 0.00 | 100% | 100% | recall | $0.01 | $0.00 |

Median total: 2.0 / 2 — Agent: $0.01  Judge: $0.00  Total: $0.01

---

### q01 — Where is X defined?

**Run 1**: score=2 file_ok=True judge=pass agent=$0.0100 judge=$0.0050 cache_read=0
  r1: read_file(path='a.py', line_start=1, line_end=5)

X is defined at [a.py:1-5]. It does the thing.

**Run 2**: score=0 file_ok=False judge=fail agent=$0.0080 judge=$0.0040 cache_read=0

X is probably somewhere, not sure. Here's a heading in the answer:

### Not a question section

More prose after a heading the model wrote itself.

### q02 — Where is Y defined?

**Run 1**: score=1 file_ok=True judge=error agent=$0.0000 judge=$0.0000 cache_read=0

"""


def test_parse_results_md_counts_rows(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(_SAMPLE_MD)
    rows = parse_results_md(str(p))
    assert len(rows) == 3  # q01 run1, q01 run2, q02 run1


def test_parse_results_md_extracts_verdicts(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(_SAMPLE_MD)
    rows = parse_results_md(str(p))
    verdicts = {(r["id"], r["run"]): r["judge_verdict"] for r in rows}
    assert verdicts[("q01", 1)] == "pass"
    assert verdicts[("q01", 2)] == "fail"
    assert verdicts[("q02", 1)] == "error"


def test_parse_results_md_answer_strips_tool_trace(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(_SAMPLE_MD)
    rows = parse_results_md(str(p))
    run1 = next(r for r in rows if r["id"] == "q01" and r["run"] == 1)
    assert "r1: read_file" not in run1["answer"]
    assert "X is defined at" in run1["answer"]


def test_parse_results_md_does_not_split_on_answer_headings(tmp_path):
    """A model's own '### Location'-style subheading inside an answer must
    not be mistaken for the next question section — the regex the run
    hit this on 2026-08-13 split a 50-question file into 393 pieces."""
    p = tmp_path / "r.md"
    p.write_text(_SAMPLE_MD)
    rows = parse_results_md(str(p))
    run2 = next(r for r in rows if r["id"] == "q01" and r["run"] == 2)
    assert "### Not a question section" in run2["answer"]
    assert "More prose after a heading" in run2["answer"]


def test_load_question_rubrics_skips_meta(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join([
        json.dumps({"_meta": "header"}),
        json.dumps({"id": "q01", "description_must_include": ["a"]}),
    ]))
    rubrics = load_question_rubrics(str(p))
    assert list(rubrics) == ["q01"]
    assert rubrics["q01"]["description_must_include"] == ["a"]


# ── stratified sampling ───────────────────────────────────────────────────────

def _row(qid, run, verdict):
    return {"id": qid, "run": run, "question": f"Q for {qid}", "judge_verdict": verdict, "answer": "..."}


def test_stratified_sample_includes_all_non_pass():
    rows = [_row("q01", 1, "fail"), _row("q02", 1, "error")]
    rows += [_row(f"q{i:02}", 1, "pass") for i in range(3, 20)]
    sample = stratified_sample(rows, n=10)
    verdicts = [r["judge_verdict"] for r in sample]
    assert verdicts.count("fail") == 1
    assert verdicts.count("error") == 1


def test_stratified_sample_prefers_distinct_questions_over_repeat_runs():
    """3 runs of the same question is less informative than 3 different
    questions -- the sampler must not fill up on one question's runs."""
    rows = [_row("q01", r, "pass") for r in (1, 2, 3)]
    rows += [_row("q02", 1, "pass"), _row("q03", 1, "pass")]
    sample = stratified_sample(rows, n=3)
    assert len({r["id"] for r in sample}) == 3


def test_stratified_sample_caps_at_n():
    rows = [_row(f"q{i:02}", 1, "pass") for i in range(60)]
    assert len(stratified_sample(rows, n=40)) == 40


def test_stratified_sample_deterministic_with_fixed_seed():
    rows = [_row(f"q{i:02}", 1, "pass" if i % 5 else "fail") for i in range(30)]
    a = stratified_sample(rows, n=10, seed=1)
    b = stratified_sample(rows, n=10, seed=1)
    assert [r["id"] for r in a] == [r["id"] for r in b]


# ── blind / key file writers ────────────────────────────────────────────────

def test_write_blind_file_hides_verdict(tmp_path):
    sample = [_row("q01", 1, "fail")]
    rubrics = {"q01": {"description_must_include": ["a"], "description_must_not_assert": ["b"]}}
    out = tmp_path / "blind.jsonl"
    write_blind_file(sample, rubrics, out)
    row = json.loads(out.read_text())
    assert "judge_verdict" not in row
    assert row["human_verdict"] is None
    assert row["sample_id"] == "q01-run1"
    assert row["description_must_include"] == ["a"]


def test_write_key_file_has_no_answer_or_question_text(tmp_path):
    """The key exists only to score against, and holds nothing that would
    let the verdict leak if someone glanced at it -- no question text, no
    answer text, just the id and the pass/fail it's keyed to."""
    sample = [_row("q01", 1, "fail")]
    out = tmp_path / "key.jsonl"
    write_key_file(sample, out)
    row = json.loads(out.read_text())
    assert set(row) == {"sample_id", "judge_verdict"}
    assert row["judge_verdict"] == "fail"


# ── Cohen's kappa ─────────────────────────────────────────────────────────────

def test_kappa_perfect_agreement():
    a = [True, False, True, False, True]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9


def test_kappa_chance_agreement_is_zero():
    # Constructed so observed agreement equals expected-by-chance agreement.
    a = [True, True, False, False]
    b = [True, False, True, False]
    # po = 2/4 = 0.5; p(a=T)=0.5, p(b=T)=0.5 -> pe = 0.5*0.5+0.5*0.5 = 0.5
    assert abs(cohens_kappa(a, b) - 0.0) < 1e-9


def test_kappa_systematic_disagreement_is_negative():
    a = [True, True, False, False]
    b = [False, False, True, True]
    assert cohens_kappa(a, b) < 0


def test_kappa_empty_is_zero():
    assert cohens_kappa([], []) == 0.0


def test_kappa_both_constant_and_equal_is_one():
    assert cohens_kappa([True, True, True], [True, True, True]) == 1.0
