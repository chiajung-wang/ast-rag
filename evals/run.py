from __future__ import annotations
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
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
    return (3.00, 15.00)


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = _price_per_mtok(model)
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def compute_score(file_ok: bool, judge_pass: bool) -> int:
    return int(file_ok) + int(judge_pass)


def check_file_ok(expected_paths: list[str], answer: str) -> bool:
    return any(path in answer for path in expected_paths)


def format_results_md(rows: list[dict], n_runs: int) -> str:
    lines = [
        "| id | question | median | var | file_ok% | judge% | tier | agent_cost | judge_cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        runs = r["runs"]
        file_pct = f"{sum(1 for x in runs if x['file_ok']) / len(runs):.0%}"
        judge_pct = f"{sum(1 for x in runs if x['judge'] == 'pass') / len(runs):.0%}"
        lines.append(
            f"| {r['id']} | {q_short} | {r['median_score']} | {r['variance']:.2f}"
            f" | {file_pct} | {judge_pct} | {r['tier']}"
            f" | ${r['agent_cost']:.4f} | ${r['judge_cost']:.4f} |"
        )
    total_agent = sum(r["agent_cost"] for r in rows)
    total_judge = sum(r["judge_cost"] for r in rows)
    total_median = sum(r["median_score"] for r in rows)
    max_total = len(rows) * 2
    lines.append(
        f"\nMedian total: {total_median:.1f} / {max_total}"
        f" — Agent: ${total_agent:.4f}  Judge: ${total_judge:.4f}"
        f"  Total: ${total_agent + total_judge:.4f}"
    )
    lines.append("\n---\n")
    for r in rows:
        lines.append(f"### {r['id']} — {r['question']}\n")
        for i, run in enumerate(r["runs"], 1):
            lines.append(
                f"**Run {i}**: score={run['score']} file_ok={run['file_ok']}"
                f" judge={run['judge']} agent=${run['agent_cost']:.4f} judge=${run['judge_cost']:.4f}"
            )
            trace = run.get("tool_trace", [])
            if trace:
                for t in trace:
                    args_str = ", ".join(f"{k}={v!r}" for k, v in t["args"].items())
                    lines.append(f"  r{t['round']}: {t['tool']}({args_str})")
            lines.append("")
            lines.append(run.get("answer", "(no answer)"))
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
    model = ChatAnthropic(model=JUDGE_MODEL, temperature=0)
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
            verdict, _ = json.JSONDecoder().raw_decode(raw)
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


def _run_once(q: dict) -> dict:
    answer = ""
    agent_cost = 0.0
    judge_cost = 0.0
    score = 0
    file_ok = False
    judge_str = "error"
    try:
        result = graph.invoke({
            "messages": [HumanMessage(content=q["question"])],
            "retrieved_chunks": [],
        })
        last_msg = result["messages"][-1]
        answer = last_msg.content
        usage = last_msg.usage_metadata or {}
        agent_cost = compute_cost(
            AGENT_MODEL,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        tool_trace = last_msg.additional_kwargs.get("tool_trace", [])
        file_ok = check_file_ok(q["expected_file_paths"], answer)
        judge_pass, judge_in, judge_out = _judge(
            q["question"],
            q["expected_file_paths"],
            q["description_must_include"],
            q.get("description_must_not_assert", []),
            answer,
        )
        judge_cost = compute_cost(JUDGE_MODEL, judge_in, judge_out)
        score = compute_score(file_ok, judge_pass)
        judge_str = "pass" if judge_pass else "fail"
    except Exception as exc:
        print(f"    ERROR: {exc!r}")
    return {
        "score": score,
        "file_ok": file_ok,
        "judge": judge_str,
        "agent_cost": agent_cost,
        "judge_cost": judge_cost,
        "answer": answer,
        "tool_trace": tool_trace,
    }


def run(
    questions_path: str = "evals/questions.jsonl",
    results_path: str = "evals/results.md",
    start: int = 1,
    end: int | None = None,
    n_runs: int = 3,
) -> None:
    with open(questions_path) as f:
        questions = [
            json.loads(line) for line in f
            if line.strip() and not json.loads(line.strip()).get("_meta")
        ]
    questions = questions[start - 1 : end]

    print(f"agent={AGENT_MODEL}  judge={JUDGE_MODEL}  temperature=0  runs={n_runs}  questions={len(questions)}")
    rows = []
    try:
        for q in questions:
            print(f"{q['id']}:")
            runs = []
            for i in range(n_runs):
                r = _run_once(q)
                runs.append(r)
                print(
                    f"  run {i + 1}: score={r['score']} file_ok={r['file_ok']}"
                    f" judge={r['judge']} agent=${r['agent_cost']:.4f} judge=${r['judge_cost']:.4f}"
                )

            scores = [r["score"] for r in runs]
            median_score = statistics.median(scores)
            variance = statistics.variance(scores) if len(scores) > 1 else 0.0
            agent_cost_total = sum(r["agent_cost"] for r in runs)
            judge_cost_total = sum(r["judge_cost"] for r in runs)
            print(
                f"  → median={median_score} var={variance:.2f}"
                f" agent=${agent_cost_total:.4f} judge=${judge_cost_total:.4f}"
            )

            rows.append({
                "id": q["id"],
                "question": q["question"],
                "median_score": median_score,
                "variance": variance,
                "runs": runs,
                "tier": q.get("tier", ""),
                "agent_cost": agent_cost_total,
                "judge_cost": judge_cost_total,
            })
            md = format_results_md(rows, n_runs)
            with open(results_path, "w", encoding="utf-8") as f:
                f.write(md)

    except KeyboardInterrupt:
        print(f"\nInterrupted — partial results written to {results_path}")
        return

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--results-dir", default="evals/results")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d-%H%M")
    agent_slug = AGENT_MODEL.split("claude-")[-1].split("-2")[0]
    judge_slug = JUDGE_MODEL.split("claude-")[-1].split("-2")[0]
    results_path = str(results_dir / f"results-{timestamp}-{agent_slug}-{judge_slug}.md")
    run(args.questions, results_path, args.start, args.end, args.runs)
