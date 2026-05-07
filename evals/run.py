from __future__ import annotations
import json
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import graph
from evals.judge_prompt import JUDGE_SYSTEM

JUDGE_MODEL = "claude-sonnet-4-6"


def compute_score(file_ok: bool, judge_pass: bool) -> int:
    return int(file_ok) + int(judge_pass)


def check_file_ok(expected_paths: list[str], answer: str) -> bool:
    return any(path in answer for path in expected_paths)


def format_results_md(rows: list[dict]) -> str:
    lines = [
        "| id | question | score | file_ok | judge | tier |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        file_sym = "✓" if r["file_ok"] else "✗"
        tier = r.get("tier", "")
        lines.append(
            f"| {r['id']} | {q_short} | {r['score']} | {file_sym} | {r['judge']} | {tier} |"
        )
    total = sum(r["score"] for r in rows)
    max_total = len(rows) * 2
    lines.append(f"\nTotal: {total} / {max_total}")
    lines.append("\n---\n")
    for r in rows:
        lines.append(f"### {r['id']} — {r['question']}\n")
        lines.append(r.get("answer", "(no answer recorded)"))
        lines.append("")
    return "\n".join(lines)


def _judge(
    question: str,
    expected_file_paths: list[str],
    must_include: list[str],
    must_not_assert: list[str],
    answer: str,
    max_retries: int = 3,
) -> bool:
    model = ChatAnthropic(model=JUDGE_MODEL)
    payload = {
        "question": question,
        "expected_file_paths": expected_file_paths,
        "description_must_include": must_include,
        "description_must_not_assert": must_not_assert,
        "model_answer": answer,
    }
    prompt = json.dumps(payload, indent=2)
    messages = [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)]
    for attempt in range(max_retries):
        try:
            response = model.invoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    line for line in raw.splitlines()
                    if not line.startswith("```")
                ).strip()
            verdict = json.loads(raw)
            return bool(verdict.get("description_correct", False))
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  judge error (attempt {attempt + 1}): {exc!r} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  judge failed after {max_retries} attempts: {exc!r} — scoring as fail")
                return False


def run(
    questions_path: str = "evals/questions.jsonl",
    results_path: str = "evals/results.md",
) -> None:
    with open(questions_path) as f:
        questions = [
            json.loads(line) for line in f
            if line.strip() and not json.loads(line.strip()).get("_meta")
        ]

    rows = []
    for q in questions:
        answer = ""
        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=q["question"])],
                "retrieved_chunks": [],
            })
            answer = result["messages"][-1].content
            file_ok = check_file_ok(q["expected_file_paths"], answer)
            judge_pass = _judge(
                q["question"],
                q["expected_file_paths"],
                q["description_must_include"],
                q.get("description_must_not_assert", []),
                answer,
            )
            score = compute_score(file_ok, judge_pass)
            judge_str = "pass" if judge_pass else "fail"
        except Exception as exc:
            print(f"{q['id']}: ERROR — {exc!r}")
            score, file_ok, judge_str, answer = 0, False, "error", ""
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "score": score,
            "file_ok": file_ok,
            "judge": judge_str,
            "tier": q.get("tier", ""),
            "answer": answer,
        })
        print(f"{q['id']}: score={score} file_ok={file_ok} judge={judge_str}")
        md = format_results_md(rows)
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    run()
