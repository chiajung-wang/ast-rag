from __future__ import annotations
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import graph
from evals.judge_prompt import JUDGE_SYSTEM
import provider

JUDGE_MODEL = provider.judge_model()
AGENT_MODEL = provider.agent_model()

# USD per million tokens, (input, output). OpenRouter passes provider pricing
# through unchanged, so these match Anthropic list prices as of 2026-08-12.
# Keys are OpenRouter slugs: a miss here falls back to the Sonnet rate and
# silently misprices every run, which is the bug task 6.1 fixed.
#
# Sonnet 5 uses dashes (claude-sonnet-5), not the dots seen on 4.5/4.6 --
# verified against openrouter.ai/anthropic/claude-sonnet-5. Its $2/$10 rate is
# an introductory price through 2026-08-31; standard is $3/$15, same as 4.6.
_PRICES: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-opus-4.7": (5.00, 25.00),
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


def total_score(rows: list[dict]) -> tuple[float, int]:
    total_median = sum(r["median_score"] for r in rows)
    max_total = sum(1 if r["tier"] == "negative" else 2 for r in rows)
    return total_median, max_total


BASELINE_PATH = Path("evals/baseline.json")


def load_baseline(path: str | Path = BASELINE_PATH) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_baseline(baseline: dict, path: str | Path = BASELINE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def baseline_key_for(questions_path: str) -> str:
    """Which baseline.json entry a question set gates against. The dev file
    is evals/questions.jsonl; anything else (currently only the held-out
    set) is "test". A question set neither of these has no baseline key and
    the gate is skipped rather than guessed at."""
    name = Path(questions_path).name
    if name == "questions.jsonl":
        return "dev"
    if name == "questions-test.jsonl":
        return "test"
    return name


def check_regression(
    total_median: float, max_total: int, key: str, baseline: dict, tolerance: float = 2.0
) -> tuple[bool, str]:
    """True/message pair, never raises and never exits -- run() stays a
    plain library function callable from tests without a stray SystemExit.
    The actual process exit code is decided by the caller in __main__."""
    entry = baseline.get(key)
    if not entry or entry.get("median_total") is None:
        return True, f"no baseline recorded for '{key}' — nothing to compare against"
    if entry.get("max_total") != max_total:
        return True, (
            f"baseline max_total={entry.get('max_total')} but this run's max_total={max_total}"
            f" — question set changed since the baseline was recorded, skipping the gate"
        )
    drop = entry["median_total"] - total_median
    if drop > tolerance:
        return False, (
            f"REGRESSION: {total_median:.1f}/{max_total} is {drop:.1f} points below"
            f" baseline {entry['median_total']}/{entry['max_total']} (tolerance {tolerance})"
        )
    return True, (
        f"{total_median:.1f}/{max_total} vs baseline {entry['median_total']}/{entry['max_total']}"
        f" — within tolerance"
    )


def check_file_ok(expected_paths: list[str], answer: str) -> bool:
    return any(path in answer for path in expected_paths)


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile, no numpy dependency for one function.
    p is 0-100. Empty input returns 0.0 rather than raising -- a tier with no
    runs shouldn't crash the report."""
    if not data:
        return 0.0
    s = sorted(data)
    if len(s) == 1:
        return s[0]
    k = (p / 100) * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def aggregate_by_tier(rows: list[dict]) -> list[dict]:
    """One row per tier: latency, tool rounds, budget exhaustion, and
    citation quality, pooled across every run of every question in that
    tier. Citation rates are pooled counts (sum stripped / sum emitted), not
    an average of per-run rates -- a run with 0 citations would otherwise
    contribute a meaningless 0% or undefined rate and skew the mean."""
    by_tier: dict[str, list[dict]] = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).extend(r["runs"])

    out = []
    for tier in sorted(by_tier):
        runs = by_tier[tier]
        latencies = [x["latency_s"] for x in runs if x.get("latency_s")]
        rounds = [x["rounds"] for x in runs]
        exhausted = sum(1 for x in runs if x.get("budget_exhausted"))
        emitted = sum(x["citation_stats"]["emitted"] for x in runs)
        stripped = sum(x["citation_stats"]["stripped"] for x in runs)
        hallucinated = sum(x["citation_stats"]["hallucinated_paths"] for x in runs)
        survived = sum(x["citation_stats"]["survived"] for x in runs)
        precise = sum(x["citation_stats"]["precise"] for x in runs)
        out.append({
            "tier": tier,
            "n_runs": len(runs),
            "p50_latency_s": percentile(latencies, 50),
            "p95_latency_s": percentile(latencies, 95),
            "mean_rounds": sum(rounds) / len(rounds) if rounds else 0.0,
            "exhausted_rate": exhausted / len(runs) if runs else 0.0,
            "strip_rate": stripped / emitted if emitted else 0.0,
            "hallucinated_path_rate": hallucinated / emitted if emitted else 0.0,
            "precision": precise / survived if survived else 0.0,
        })
    return out


def format_tier_report_md(rows: list[dict]) -> str:
    agg = aggregate_by_tier(rows)
    lines = [
        "\n## Per-tier instrumentation\n",
        "| tier | n | p50 latency | p95 latency | mean rounds | exhausted% | citation precision | strip% | hallucinated-path% |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in agg:
        lines.append(
            f"| {a['tier']} | {a['n_runs']} | {a['p50_latency_s']:.1f}s | {a['p95_latency_s']:.1f}s"
            f" | {a['mean_rounds']:.1f} | {a['exhausted_rate']:.0%}"
            f" | {a['precision']:.0%} | {a['strip_rate']:.0%} | {a['hallucinated_path_rate']:.0%} |"
        )
    return "\n".join(lines)


def format_results_md(rows: list[dict], n_runs: int) -> str:
    lines = [
        "| id | question | median | var | file_ok% | judge% | tier | agent_cost | judge_cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q_short = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
        runs = r["runs"]
        is_negative = r["tier"] == "negative"
        file_pct = "n/a" if is_negative else f"{sum(1 for x in runs if x['file_ok']) / len(runs):.0%}"
        judge_pct = f"{sum(1 for x in runs if x['judge'] == 'pass') / len(runs):.0%}"
        lines.append(
            f"| {r['id']} | {q_short} | {r['median_score']} | {r['variance']:.2f}"
            f" | {file_pct} | {judge_pct} | {r['tier']}"
            f" | ${r['agent_cost']:.4f} | ${r['judge_cost']:.4f} |"
        )
    total_agent = sum(r["agent_cost"] for r in rows)
    total_judge = sum(r["judge_cost"] for r in rows)
    total_median, max_total = total_score(rows)
    lines.append(
        f"\nMedian total: {total_median:.1f} / {max_total}"
        f" — Agent: ${total_agent:.4f}  Judge: ${total_judge:.4f}"
        f"  Total: ${total_agent + total_judge:.4f}"
    )
    lines.append(format_tier_report_md(rows))
    lines.append("\n---\n")
    for r in rows:
        lines.append(f"### {r['id']} — {r['question']}\n")
        for i, run in enumerate(r["runs"], 1):
            lines.append(
                f"**Run {i}**: score={run['score']} file_ok={run['file_ok']}"
                f" judge={run['judge']} agent=${run['agent_cost']:.4f} judge=${run['judge_cost']:.4f}"
                f" cache_read={run.get('cache_read_tokens', 0)}"
                f" latency={run.get('latency_s', 0.0):.1f}s rounds={run.get('rounds', 0)}"
            )
            cs = run.get("citation_stats") or {}
            if cs.get("emitted"):
                lines.append(
                    f"  citations: emitted={cs['emitted']} stripped={cs['stripped']}"
                    f" hallucinated_paths={cs['hallucinated_paths']} precision={cs['precise']}/{cs['survived']}"
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
    model_name: str | None = None,
) -> tuple[bool, int, int]:
    # model_name lets judge_validation.py run the same rubric through a
    # second model for a cross-judge agreement check, without touching the
    # live JUDGE_MODEL config any real eval run uses.
    model = ChatOpenAI(
        model=model_name or JUDGE_MODEL,
        temperature=0,
        base_url=provider.BASE_URL,
        api_key=provider.api_key(),
    )
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
    tool_trace: list[dict] = []
    cache_read = 0
    latency_s = 0.0
    budget_exhausted = False
    citation_stats = {"emitted": 0, "stripped": 0, "hallucinated_paths": 0, "precise": 0, "survived": 0}
    try:
        t0 = time.perf_counter()
        result = graph.invoke({
            "messages": [HumanMessage(content=q["question"])],
            "retrieved_chunks": [],
        })
        latency_s = time.perf_counter() - t0
        last_msg = result["messages"][-1]
        answer = last_msg.content
        usage = last_msg.usage_metadata or {}
        agent_cost = compute_cost(
            AGENT_MODEL,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        tool_trace = last_msg.additional_kwargs.get("tool_trace", [])
        cache_read = last_msg.additional_kwargs.get("cache_read_tokens", 0)
        budget_exhausted = last_msg.additional_kwargs.get("budget_exhausted", False)
        citation_stats = last_msg.additional_kwargs.get("citation_stats", citation_stats)
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
        if judge_pass:
            judge_str = "pass"
        elif budget_exhausted:
            judge_str = "fail/exhausted"
        else:
            judge_str = "fail"
    except Exception as exc:
        print(f"    ERROR: {exc!r}")
    rounds = len({t["round"] for t in tool_trace}) if tool_trace else 0
    return {
        "cache_read_tokens": cache_read,
        "score": score,
        "file_ok": file_ok,
        "judge": judge_str,
        "agent_cost": agent_cost,
        "judge_cost": judge_cost,
        "answer": answer,
        "tool_trace": tool_trace,
        "latency_s": latency_s,
        "rounds": rounds,
        "budget_exhausted": budget_exhausted,
        "citation_stats": citation_stats,
    }


def run(
    questions_path: str = "evals/questions.jsonl",
    results_path: str = "evals/results.md",
    start: int = 1,
    end: int | None = None,
    n_runs: int = 3,
    baseline_path: str | Path = BASELINE_PATH,
) -> list[dict]:
    """Returns the per-question rows. Never exits or raises on a regression
    -- that decision belongs to the caller (see __main__), so run() stays
    callable as a plain function from tests and from evals.retrieval_eval-
    style scripts without a surprise SystemExit. baseline_path is a
    parameter (not the bare module constant) so tests can point it at an
    isolated file instead of silently reading the real repo baseline.json."""
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
        return rows

    print(f"\nResults written to {results_path}")

    if rows:
        total_median, max_total = total_score(rows)
        baseline = load_baseline(baseline_path)
        key = baseline_key_for(questions_path)
        _passed, message = check_regression(total_median, max_total, key, baseline)
        print(message)

    return rows


def main(argv: list[str] | None = None) -> int:
    """Returns the process exit code rather than calling sys.exit itself, so
    tests can assert on it directly instead of catching SystemExit."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--results-dir", default="evals/results")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write this run's score as the new evals/baseline.json entry instead of gating on it")
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d-%H%M")
    agent_slug = AGENT_MODEL.split("/")[-1]
    judge_slug = JUDGE_MODEL.split("/")[-1]
    # Name the question set in the filename. A dev score and a held-out score
    # are different claims and must not be mistaken for each other on disk.
    set_slug = Path(args.questions).stem.replace("questions-", "").replace("questions", "dev")
    results_path = str(results_dir / f"results-{timestamp}-{set_slug}-{agent_slug}-{judge_slug}.md")
    result_rows = run(args.questions, results_path, args.start, args.end, args.runs)

    if not result_rows:
        return 0  # interrupted before any question completed -- nothing to gate

    total_median, max_total = total_score(result_rows)
    key = baseline_key_for(args.questions)

    if args.update_baseline:
        baseline = load_baseline()
        baseline[key] = {"median_total": total_median, "max_total": max_total, "n_runs": args.runs}
        baseline["updated"] = datetime.now().strftime("%Y-%m-%d")
        save_baseline(baseline)
        print(f"baseline['{key}'] updated to {total_median}/{max_total} (n_runs={args.runs})")
        return 0

    # run() already printed the check_regression message once; this repeats
    # the cheap, pure comparison only to decide the process exit code.
    baseline = load_baseline()
    passed, _message = check_regression(total_median, max_total, key, baseline)
    return 0 if passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
