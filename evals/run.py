from __future__ import annotations
import json
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import graph

JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_SYSTEM = (
    "You are an evaluation judge. Given a question, a ground truth answer, "
    "and an agent answer, respond with exactly 'pass' if the agent answer is "
    "substantially correct, or 'fail' otherwise. No explanation."
)


def compute_score(file_ok: bool, judge_pass: bool) -> int:
    return int(file_ok) + int(judge_pass)


def check_file_ok(expected_path: str, answer: str) -> bool:
    return expected_path in answer


def format_results_md(rows: list[dict]) -> str:
    lines = [
        "| id | question | score | file_ok | judge |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        file_sym = "✓" if r["file_ok"] else "✗"
        lines.append(f"| {r['id']} | {q_short} | {r['score']} | {file_sym} | {r['judge']} |")
    total = sum(r["score"] for r in rows)
    max_total = len(rows) * 2
    lines.append(f"\nTotal: {total} / {max_total}")
    return "\n".join(lines)


def _judge(question: str, ground_truth: str, answer: str) -> bool:
    model = ChatAnthropic(model=JUDGE_MODEL)
    prompt = (
        f"Question: {question}\n\n"
        f"Ground truth: {ground_truth}\n\n"
        f"Agent answer: {answer}"
    )
    response = model.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=prompt)])
    return response.content.strip().lower() == "pass"


def run(
    questions_path: str = "evals/questions.jsonl",
    results_path: str = "evals/results.md",
) -> None:
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    rows = []
    for q in questions:
        result = graph.invoke({
            "messages": [HumanMessage(content=q["question"])],
            "retrieved_chunks": [],
        })
        answer = result["messages"][-1].content
        file_ok = check_file_ok(q["expected_file_path"], answer)
        judge_pass = _judge(q["question"], q["ground_truth_answer"], answer)
        score = compute_score(file_ok, judge_pass)
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "score": score,
            "file_ok": file_ok,
            "judge": "pass" if judge_pass else "fail",
        })
        print(f"{q['id']}: score={score} file_ok={file_ok} judge={'pass' if judge_pass else 'fail'}")

    md = format_results_md(rows)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    run()
