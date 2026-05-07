from __future__ import annotations
import json
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
    return "\n".join(lines)


def _judge(
    question: str,
    must_include: list[str],
    must_not_assert: list[str],
    answer: str,
) -> bool:
    model = ChatAnthropic(model=JUDGE_MODEL)
    must_include_str = "\n".join(f"- {item}" for item in must_include)
    must_not_str = "\n".join(f"- {item}" for item in must_not_assert) if must_not_assert else "(none)"
    prompt = (
        f"Question: {question}\n\n"
        f"MUST_INCLUDE (all required):\n{must_include_str}\n\n"
        f"MUST_NOT_ASSERT (automatic fail if present):\n{must_not_str}\n\n"
        f"Agent answer:\n{answer}"
    )
    response = model.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)])
    return response.content.strip().lower() == "pass"


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
        result = graph.invoke({
            "messages": [HumanMessage(content=q["question"])],
            "retrieved_chunks": [],
        })
        answer = result["messages"][-1].content
        file_ok = check_file_ok(q["expected_file_paths"], answer)
        judge_pass = _judge(
            q["question"],
            q["description_must_include"],
            q.get("description_must_not_assert", []),
            answer,
        )
        score = compute_score(file_ok, judge_pass)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "score": score,
            "file_ok": file_ok,
            "judge": "pass" if judge_pass else "fail",
            "tier": q.get("tier", ""),
        })
        print(f"{q['id']}: score={score} file_ok={file_ok} judge={'pass' if judge_pass else 'fail'}")

    md = format_results_md(rows)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    run()
