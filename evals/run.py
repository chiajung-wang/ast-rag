from __future__ import annotations
import json
import os
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import graph
from evals.judge_prompt import JUDGE_SYSTEM

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")

_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7": (15.00, 75.00),
}

def _price_per_mtok(model: str) -> tuple[float, float]:
    for prefix, rates in _PRICES.items():
        if model.startswith(prefix):
            return rates
    return (3.00, 15.00)  # fallback: sonnet rates


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = _price_per_mtok(model)
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def compute_score(file_ok: bool, judge_pass: bool) -> int:
    return int(file_ok) + int(judge_pass)


def check_file_ok(expected_paths: list[str], answer: str) -> bool:
    return any(path in answer for path in expected_paths)


def format_results_md(rows: list[dict]) -> str:
    lines = [
        "| id | question | score | file_ok | judge | tier | cost |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        file_sym = "✓" if r["file_ok"] else "✗"
        tier = r.get("tier", "")
        cost = r.get("cost", 0.0)
        lines.append(
            f"| {r['id']} | {q_short} | {r['score']} | {file_sym} | {r['judge']} | {tier} | ${cost:.4f} |"
        )
    total = sum(r["score"] for r in rows)
    max_total = len(rows) * 2
    total_cost = sum(r.get("cost", 0.0) for r in rows)
    lines.append(f"\nTotal: {total} / {max_total} — Cost: ${total_cost:.4f}")
    lines.append("\n---\n")
    for r in rows:
        cost = r.get("cost", 0.0)
        lines.append(f"### {r['id']} — {r['question']} (${cost:.4f})\n")
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
) -> tuple[bool, int, int]:
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
            usage = response.usage_metadata or {}
            return (
                bool(verdict.get("description_correct", False)),
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  judge error (attempt {attempt + 1}): {exc!r} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  judge failed after {max_retries} attempts: {exc!r} — scoring as fail")
                return False, 0, 0


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
        cost = 0.0
        try:
            result = graph.invoke({
                "messages": [HumanMessage(content=q["question"])],
                "retrieved_chunks": [],
            })
            answer = result["messages"][-1].content
            agent_input_tok = sum(len(str(m.content)) for m in result["messages"][:-1]) // 4
            agent_output_tok = len(answer) // 4
            agent_cost = compute_cost(AGENT_MODEL, agent_input_tok, agent_output_tok)

            file_ok = check_file_ok(q["expected_file_paths"], answer)
            judge_pass, judge_in, judge_out = _judge(
                q["question"],
                q["expected_file_paths"],
                q["description_must_include"],
                q.get("description_must_not_assert", []),
                answer,
            )
            judge_cost = compute_cost(JUDGE_MODEL, judge_in, judge_out)
            cost = agent_cost + judge_cost
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
            "cost": cost,
        })
        print(f"{q['id']}: score={score} file_ok={file_ok} judge={judge_str} cost=${cost:.4f}")
        md = format_results_md(rows)
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    run()
