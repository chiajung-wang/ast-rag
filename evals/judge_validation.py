"""How much can the LLM judge be trusted? Two independent checks.

1. Human agreement (Cohen's kappa). `evals/judge_prompt.py` drives a binary
   pass/fail decision and nothing has ever measured whether it agrees with a
   person. This module samples 40 (question, answer) pairs from a completed
   eval run, hides the judge's verdict, and lets a human label them blind.
   Kappa is then computed between the human labels and the hidden verdicts.

   The blind step is the whole point and cannot be automated or drafted by
   an agent — an agent-drafted "independent" opinion is not independent of
   the system being measured. `sample` writes the blind file; a human fills
   in every `human_verdict`; `score` joins it against the hidden key.

2. Cross-judge agreement (fully automated, no bias concern). The same 40
   answers are re-judged by a second model. This measures how much of the
   judge's verdict is judge-specific noise versus something a different
   model would also conclude — a real error term in every published score.

Usage:
    python -m evals.judge_validation sample --n 40
    # ... hand-label evals/judge_validation_blind.jsonl, one human_verdict
    # field per row, pass or fail, without looking at any judge output ...
    python -m evals.judge_validation score
    python -m evals.judge_validation cross-judge --model anthropic/claude-haiku-4.5
"""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path

BLIND_PATH = Path("evals/judge_validation_blind.jsonl")
KEY_PATH = Path("evals/.judge_validation_key.jsonl")  # gitignored: holds the verdict being validated
REPORT_PATH = Path("evals/judge_validation.jsonl")

_SECTION_RE = re.compile(r"^### (q\d+|t\d+) — ", re.M)
_RUN_RE = re.compile(r"^\*\*Run (\d+)\*\*: (.+)$", re.M)
_VERDICT_RE = re.compile(r"judge=(pass|fail/exhausted|fail|error)")
_TRACE_LINE_RE = re.compile(r"^  r\d+: .+$\n?", re.M)


def parse_results_md(path: str) -> list[dict]:
    """Every (question, run) row in a results file, with its judge verdict
    and the full answer text. The markdown has no structured form, so this
    is a regex parser tied to format_results_md's exact template — verified
    against a real file (150/150 rows) before being trusted here."""
    text = Path(path).read_text(encoding="utf-8")
    sections = _SECTION_RE.split(text)[1:]  # [id, body, id, body, ...]

    rows = []
    for qid, body in zip(sections[0::2], sections[1::2]):
        question_text, rest = body.split("\n", 1)
        question_text = question_text.strip()
        parts = _RUN_RE.split(rest)[1:]  # [run_num, header, block, run_num, header, block, ...]
        for run_num, header, block in zip(parts[0::3], parts[1::3], parts[2::3]):
            m = _VERDICT_RE.search(header)
            if not m:
                continue  # judge=error rows with no completed verdict aren't gradeable
            answer = _TRACE_LINE_RE.sub("", block).strip()
            rows.append({
                "id": qid,
                "run": int(run_num),
                "question": question_text,
                "judge_verdict": m.group(1),
                "answer": answer,
            })
    return rows


def load_question_rubrics(path: str = "evals/questions.jsonl") -> dict[str, dict]:
    rubrics = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("_meta"):
            continue
        rubrics[q["id"]] = q
    return rubrics


def stratified_sample(rows: list[dict], n: int, seed: int = 6) -> list[dict]:
    """All non-pass rows (rare, and exactly where a judge is most likely to
    be wrong), topped up with one pass row per distinct question for
    diversity, rather than 3 near-identical runs of the same question.

    Rationale for oversampling fail/error: with a judge that passes ~93% of
    answers, a plain random sample of 40 would carry only 2-3 fail cases --
    not enough to say anything about the judge's false-positive rate, which
    is the direction that matters most (a judge that rubber-stamps wrong
    answers is worse than one that's occasionally too strict).
    """
    rng = random.Random(seed)
    non_pass = [r for r in rows if r["judge_verdict"] != "pass"]
    passing = [r for r in rows if r["judge_verdict"] == "pass"]

    rng.shuffle(passing)
    seen_ids = set()
    pass_diverse = []
    for r in passing:
        if r["id"] not in seen_ids:
            pass_diverse.append(r)
            seen_ids.add(r["id"])

    need = max(0, n - len(non_pass))
    sample = non_pass + pass_diverse[:need]
    rng.shuffle(sample)
    return sample[:n]


def write_blind_file(sample: list[dict], rubrics: dict[str, dict], path: Path = BLIND_PATH) -> None:
    lines = []
    for row in sample:
        rubric = rubrics.get(row["id"], {})
        lines.append(json.dumps({
            "sample_id": f"{row['id']}-run{row['run']}",
            "question": row["question"],
            "description_must_include": rubric.get("description_must_include", []),
            "description_must_not_assert": rubric.get("description_must_not_assert", []),
            "answer": row["answer"],
            "human_verdict": None,  # fill with "pass" or "fail" -- apply the same rubric the judge uses
        }, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_key_file(sample: list[dict], path: Path = KEY_PATH) -> None:
    lines = [
        json.dumps({"sample_id": f"{row['id']}-run{row['run']}", "judge_verdict": row["judge_verdict"]})
        for row in sample
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Standard 2x2 Cohen's kappa. No new dependency for one formula."""
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a_true = sum(a) / n
    p_b_true = sum(b) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe >= 1.0:
        return 1.0  # both raters constant and equal -- no disagreement possible
    return (po - pe) / (1 - pe)


def do_sample(args: argparse.Namespace) -> None:
    rows = parse_results_md(args.results)
    sample = stratified_sample(rows, args.n)
    rubrics = load_question_rubrics(args.questions)
    write_blind_file(sample, rubrics)
    write_key_file(sample)
    non_pass = sum(1 for r in sample if r["judge_verdict"] != "pass")
    print(f"Sampled {len(sample)} rows ({non_pass} non-pass, {len(sample) - non_pass} pass) from {args.results}")
    print(f"Blind file for labeling: {BLIND_PATH}")
    print(f"Verdict key (do not open until done labeling): {KEY_PATH}")
    print(f"\nLabel every human_verdict field as \"pass\" or \"fail\" by applying the same "
          f"rubric the judge uses (description_must_include / description_must_not_assert), "
          f"then run: python -m evals.judge_validation score")


def do_score(args: argparse.Namespace) -> None:
    blind = [json.loads(l) for l in BLIND_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = {
        json.loads(l)["sample_id"]: json.loads(l)["judge_verdict"]
        for l in KEY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    }

    unlabeled = [r["sample_id"] for r in blind if r["human_verdict"] not in ("pass", "fail")]
    if unlabeled:
        print(f"{len(unlabeled)} rows still unlabeled: {unlabeled[:5]}{'...' if len(unlabeled) > 5 else ''}")
        return

    human = [r["human_verdict"] == "pass" for r in blind]
    judge = [key[r["sample_id"]] == "pass" for r in blind]
    kappa = cohens_kappa(human, judge)
    agree = sum(1 for h, j in zip(human, judge) if h == j) / len(blind)

    report = []
    for r, h, j in zip(blind, human, judge):
        report.append({**r, "judge_verdict": key[r["sample_id"]]})
    REPORT_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in report) + "\n", encoding="utf-8")

    print(f"n={len(blind)}  raw agreement={agree:.1%}  Cohen's kappa={kappa:.3f}")
    if kappa >= 0.8:
        print("kappa >= 0.8: judge agrees with human labeling at a level that supports the eval.")
    elif kappa < 0.6:
        print("kappa < 0.6: judge disagreement with human labeling is high enough to need attention.")
    else:
        print("kappa in [0.6, 0.8): moderate agreement, worth a closer look at the disagreements below.")

    disagreements = [(r, h, j) for r, h, j in zip(blind, human, judge) if h != j]
    if disagreements:
        print(f"\n{len(disagreements)} disagreements:")
        for r, h, _j in disagreements:
            print(f"  {r['sample_id']}: human={'pass' if h else 'fail'} judge={key[r['sample_id']]}")
    print(f"\nFull report written to {REPORT_PATH}")


def do_cross_judge(args: argparse.Namespace) -> None:
    """Automated -- no human bias concern, this is just a second model."""
    from evals.run import _judge, compute_cost

    blind = [json.loads(l) for l in BLIND_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rubrics = load_question_rubrics(args.questions)
    rows = parse_results_md(args.results)
    by_id = {f"{r['id']}-run{r['run']}": r for r in rows}
    key = {
        json.loads(l)["sample_id"]: json.loads(l)["judge_verdict"]
        for l in KEY_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
    }

    agree = 0
    total_cost = 0.0
    for entry in blind:
        sid = entry["sample_id"]
        row = by_id[sid]
        rubric = rubrics.get(row["id"], {})
        cross_pass, tin, tout = _judge(
            row["question"], rubric.get("expected_file_paths", []),
            rubric.get("description_must_include", []),
            rubric.get("description_must_not_assert", []),
            row["answer"], model_name=args.model,
        )
        total_cost += compute_cost(args.model, tin, tout)
        original_pass = key[sid] == "pass"
        if cross_pass == original_pass:
            agree += 1
        else:
            print(f"  {sid}: original judge={key[sid]}  {args.model}={'pass' if cross_pass else 'fail'}")

    n = len(blind)
    print(f"\ncross-judge agreement: {agree}/{n} ({agree/n:.1%})  cost=${total_cost:.4f}")
    print(f"Disagreement rate ({(n - agree)/n:.1%}) is judge-specific noise in every published score.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="draw the blind sample from a completed results file")
    p_sample.add_argument("--results", required=True, help="path to a results-*.md file")
    p_sample.add_argument("--questions", default="evals/questions.jsonl")
    p_sample.add_argument("--n", type=int, default=40)
    p_sample.set_defaults(func=do_sample)

    p_score = sub.add_parser("score", help="compute kappa once the blind file is hand-labeled")
    p_score.set_defaults(func=do_score)

    p_cross = sub.add_parser("cross-judge", help="re-judge the same sample with a second model")
    p_cross.add_argument("--results", required=True)
    p_cross.add_argument("--questions", default="evals/questions.jsonl")
    p_cross.add_argument("--model", required=True)
    p_cross.set_defaults(func=do_cross_judge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
