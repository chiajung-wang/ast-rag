"""Retrieval-layer eval: score the retriever alone, with no LLM in the loop.

The end-to-end eval in `run.py` reports one number per question, so a failure
could come from retrieval or from generation and nothing separates the two.
This module grades retrieval against hand-written gold symbols, and compares
the four retriever configurations against each other.

No LLM calls. The only network cost is one embedding per question, cached on
disk, so a full ablation is effectively free after the first run.

Run via: python -m evals.retrieval_eval  (or: make eval-retrieval)
"""
from __future__ import annotations
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from storage.chunk import Chunk

CACHE_PATH = Path("evals/.embedding_cache.json")


# ── gold matching ─────────────────────────────────────────────────────────────

def load_questions(path: str) -> list[dict]:
    """Load questions, skipping _meta lines and the negative tier.

    Negative-tier questions have no correct chunk to retrieve, so they carry
    no signal here. `run.py` still grades them end to end.
    """
    questions = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("_meta") or not q.get("expected_symbols"):
            continue
        questions.append(q)
    return questions


def is_hit(chunk: Chunk, expected_symbols: Iterable[str], expected_file_paths: Iterable[str]) -> bool:
    """A chunk counts when its symbol is expected AND it sits in an expected file.

    The file constraint matters: symbol names collide heavily in this corpus
    (`invoke` names 23 chunks), so matching on symbol alone would credit a
    chunk from an unrelated module.
    """
    if chunk.symbol_name not in set(expected_symbols):
        return False
    paths = set(expected_file_paths)
    return not paths or chunk.file_path in paths


def hits(chunks: list[Chunk], expected_symbols: list[str], expected_file_paths: list[str]) -> list[bool]:
    return [is_hit(c, expected_symbols, expected_file_paths) for c in chunks]


# ── metrics ───────────────────────────────────────────────────────────────────
#
# Binary relevance. Gold is disjunctive: several symbols can answer one
# question, and finding any of them means retrieval succeeded.

def recall_at_k(all_hits: list[list[bool]], k: int) -> float:
    """Fraction of questions with at least one hit in the top k.

    Strictly this is a hit rate. Total relevant chunks per question is
    unknown, so a true recall denominator does not exist here.
    """
    if not all_hits:
        return 0.0
    return sum(any(h[:k]) for h in all_hits) / len(all_hits)


def mrr(all_hits: list[list[bool]]) -> float:
    """Mean reciprocal rank of the first hit. A question with no hit scores 0."""
    if not all_hits:
        return 0.0
    total = 0.0
    for h in all_hits:
        for i, hit in enumerate(h):
            if hit:
                total += 1.0 / (i + 1)
                break
    return total / len(all_hits)


def ndcg_at_k(all_hits: list[list[bool]], k: int) -> float:
    """nDCG with binary relevance.

    The ideal ranking puts every hit that was found at the top of the list.
    Total relevant is unknown, so the ideal is built from the hits actually
    retrieved. A question with no hit scores 0.
    """
    if not all_hits:
        return 0.0
    total = 0.0
    for h in all_hits:
        top = h[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, hit in enumerate(top) if hit)
        n_rel = sum(top)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
        total += dcg / idcg if idcg else 0.0
    return total / len(all_hits)


# ── embedding cache ───────────────────────────────────────────────────────────

def load_cache() -> dict[str, list[float]]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def make_cached_embedder(cache: dict[str, list[float]]) -> Callable[[str], list[float]]:
    """Embed once per question text. The ablation reuses it across configs."""
    from retrieval.pipeline import _embed_query as _real_embed

    def embed(query: str) -> list[float]:
        if query not in cache:
            cache[query] = _real_embed(query)
        return cache[query]

    return embed


# ── configurations ────────────────────────────────────────────────────────────

def run_config(
    name: str,
    retriever: Callable[[str], list[Chunk]],
    questions: list[dict],
    k: int = 5,
) -> dict:
    all_hits = [
        hits(retriever(q["question"]), q["expected_symbols"], q["expected_file_paths"])
        for q in questions
    ]
    return {
        "name": name,
        "recall@5": recall_at_k(all_hits, 5),
        "recall@10": recall_at_k(all_hits, 10),
        "mrr": mrr(all_hits),
        "ndcg@5": ndcg_at_k(all_hits, 5),
        "misses": [q["id"] for q, h in zip(questions, all_hits) if not any(h[:k])],
    }


def build_configs(k: int = 5) -> dict[str, Callable[[str], list[Chunk]]]:
    """The four retriever configurations the ablation compares."""
    from retrieval import pipeline
    from retrieval.rrf import rrf

    def bm25_only(query: str) -> list[Chunk]:
        return pipeline._get_bm25().search(query, k=10)

    def dense_only(query: str) -> list[Chunk]:
        return pipeline._get_db().vector_search(pipeline._embed_query(query), k=10)

    def rrf_hybrid(query: str) -> list[Chunk]:
        bm = pipeline._get_bm25().search(query, k=10)
        dn = pipeline._get_db().vector_search(pipeline._embed_query(query), k=10)
        return rrf(bm, dn, top_n=10)

    def rrf_plus_symbol(query: str) -> list[Chunk]:
        # k=10 so recall@10 is comparable with the other rows. Production
        # calls this with k=5; the first 5 entries are identical either way.
        return pipeline.retrieve(query, k=10)

    return {
        "BM25 only": bm25_only,
        "Dense only": dense_only,
        "RRF hybrid": rrf_hybrid,
        "RRF + symbol pre-check": rrf_plus_symbol,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def names_own_symbol(question: dict) -> bool:
    """True when the question text contains one of its own gold symbols.

    The symbol pre-check does exact name lookup, so it is favored on these.
    Reporting the split keeps that confound attached to the number.
    """
    text = question["question"].lower()
    return any(s.lower() in text for s in question["expected_symbols"])


def split_by_phrasing(rows: list[dict], questions: list[dict]) -> list[dict]:
    """Per-config recall@5 split by whether the question names its gold symbol."""
    naming = {q["id"] for q in questions if names_own_symbol(q)}
    not_naming = {q["id"] for q in questions} - naming
    out = []
    for r in rows:
        missed = set(r["misses"])
        out.append({
            "name": r["name"],
            "naming_hit": len(naming - missed),
            "naming_total": len(naming),
            "other_hit": len(not_naming - missed),
            "other_total": len(not_naming),
        })
    return out


def format_ablation_md(rows: list[dict], n_questions: int, questions: list[dict] | None = None) -> str:
    lines = [
        f"# Retrieval ablation — {n_questions} questions, gold symbols hand-labeled",
        "",
        "| configuration | recall@5 | recall@10 | MRR | nDCG@5 |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['recall@5']:.1%} | {r['recall@10']:.1%} "
            f"| {r['mrr']:.3f} | {r['ndcg@5']:.3f} |"
        )
    lines.append("")
    lines.append("Binary relevance. A question counts as retrieved when any expected symbol")
    lines.append("appears in the top k, in an expected file. Negative-tier questions are")
    lines.append("excluded — they have no correct chunk to retrieve.")
    if questions:
        splits = split_by_phrasing(rows, questions)
        naming_total = splits[0]["naming_total"]
        other_total = splits[0]["other_total"]
        lines.append("")
        lines.append("## Confound — does the question name its own answer?")
        lines.append("")
        lines.append(
            f"{naming_total} of {naming_total + other_total} questions contain a gold symbol "
            "name verbatim. The symbol pre-check does exact name lookup, so it is "
            "structurally favored on those. recall@5 split both ways:"
        )
        lines.append("")
        lines.append("| configuration | names the symbol | does not name it |")
        lines.append("|---|---|---|")
        for s in splits:
            lines.append(
                f"| {s['name']} | {s['naming_hit']}/{s['naming_total']} "
                f"| {s['other_hit']}/{s['other_total']} |"
            )
        lines.append("")
        lines.append("A question set drawn mostly from \"Where is X defined?\" cannot "
                     "separate a good retriever from a good string match. Task 6.4 adds "
                     "held-out questions phrased without the symbol name.")

    lines.append("")
    lines.append("## Misses at k=5")
    lines.append("")
    for r in rows:
        missed = ", ".join(r["misses"]) if r["misses"] else "none"
        lines.append(f"- **{r['name']}**: {missed}")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    from retrieval import pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--results-dir", default="evals/results")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"{len(questions)} questions with gold symbols")

    cache = load_cache()
    pipeline._embed_query = make_cached_embedder(cache)

    rows = []
    for name, retriever in build_configs().items():
        row = run_config(name, retriever, questions)
        rows.append(row)
        print(
            f"  {name:24} recall@5={row['recall@5']:.1%}  recall@10={row['recall@10']:.1%}"
            f"  MRR={row['mrr']:.3f}  nDCG@5={row['ndcg@5']:.3f}"
        )
    save_cache(cache)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"ablation-{datetime.now().strftime('%m%d-%H%M')}.md"
    out.write_text(format_ablation_md(rows, len(questions), questions), encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
