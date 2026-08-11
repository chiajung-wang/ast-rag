# Task 6.3 — Retrieval Metrics and Ablation

## Goal

Measure the retriever on its own. Today the eval scores answers end to end, so a failure could come from retrieval or from generation, and nothing separates them. Then compare hybrid retrieval plus RRF against each single leg.

*(The original wording was "prove that hybrid retrieval plus RRF beats each single leg." That presupposed the answer, and the measurement refuted it. Corrected after the run, and recorded here rather than quietly reworded.)*

This task produces the strongest artifact in the milestone. It converts the architecture from an asserted design into a measured result.

## Acceptance Criteria

- [x] Every non-negative question in `evals/questions.jsonl` carries an `expected_symbols` list.
- [x] `evals/retrieval_eval.py` reports recall@5, recall@10, MRR, and nDCG@5 over the question set.
- [x] The retrieval eval runs with no LLM call and finishes in under 60 seconds.
- [x] `make eval-retrieval` runs it.
- [x] The ablation table compares 4 configurations and lands in `evals/results/`.
- [x] `README.md` shows the ablation table.
- [x] `make check` passes.

## Design

### Gold labels

`expected_file_paths` already exists. A file is a coarse label, because one file holds many chunks. Add `expected_symbols`: the list of symbol names that a correct retrieval must surface.

```json
{"id": "r-01", "question": "Where is RunnableSequence defined?",
 "expected_file_paths": ["runnables/base.py"],
 "expected_symbols": ["RunnableSequence"],
 "tier": "recall", "subsystem": "runnables"}
```

A retrieved chunk counts as a hit when `chunk.symbol_name` appears in `expected_symbols`. Negative-tier questions carry an empty list and drop out of the metrics.

Label the 34 questions by hand. Use `python query.py "<question>"` to see what retrieval returns, but set the label from the source, not from the output. A label read off the current output measures nothing.

### Metrics

| Metric | Definition |
|---|---|
| recall@k | Fraction of questions where at least one expected symbol appears in the top k |
| MRR | Mean of `1 / rank` for the first expected symbol, and 0 when none appears |
| nDCG@5 | Standard nDCG with binary relevance over the top 5 |

Report recall@5 and recall@10. The gap between them shows how much a reranker could win, and it justifies the decision to leave a reranker out of scope.

### Ablation

Four configurations over the same questions:

| Configuration | How to run it |
|---|---|
| BM25 only | `BM25Index.search(query, k=5)` |
| Dense only | `db.vector_search(embedding, k=5)` |
| RRF hybrid | `rrf(bm25_10, dense_10, top_n=5)` |
| RRF + symbol pre-check | the current `retrieve()` |

Output one Markdown table with recall@5, MRR, and nDCG@5 per row. Write it to `evals/results/ablation-<mmdd-hhmm>.md` and copy the table into `README.md`.

The fourth row also answers a separate question: does the symbol pre-check earn the complexity it adds? Task 6.1 fixed two defects in that path. If the row does not beat plain RRF, say so in the README.

### Module shape

New file `evals/retrieval_eval.py`:

```python
def load_questions(path: str) -> list[dict]: ...
def hits(chunks: list[Chunk], expected_symbols: list[str]) -> list[bool]: ...
def recall_at_k(all_hits: list[list[bool]], k: int) -> float: ...
def mrr(all_hits: list[list[bool]]) -> float: ...
def ndcg_at_k(all_hits: list[list[bool]], k: int) -> float: ...
def run_config(name: str, retriever: Callable[[str], list[Chunk]], questions: list[dict]) -> dict: ...
def format_ablation_md(rows: list[dict]) -> str: ...
```

Each metric function takes plain lists of booleans, so unit tests need no database.

Cache the query embeddings on disk. The ablation runs each question through dense retrieval three times. One embedding per question is enough, and the cache keeps a full ablation under a cent.

## Files

- `evals/questions.jsonl` — add `expected_symbols` to 33 questions
- `evals/retrieval_eval.py` — new
- `tests/test_retrieval_eval.py` — new, metric unit tests with hand-built hit lists
- `Makefile` — add `eval-retrieval`
- `README.md` — add the ablation table

## Steps

- [x] Add `expected_symbols` to every non-negative question in `evals/questions.jsonl`.
- [x] Write the metric functions in `evals/retrieval_eval.py`.
- [x] Add unit tests for recall@k, MRR, and nDCG@5 against hand-built hit lists with known answers.
- [x] Add the embedding cache, keyed by the question text.
- [x] Write `run_config` and wire the 4 retriever configurations.
- [x] Write `format_ablation_md` and the `__main__` entry point.
- [x] Add `eval-retrieval` to the `Makefile`.
- [x] Run the ablation and write the result to `evals/results/`.
- [x] Copy the table into `README.md` with one sentence that states which configuration wins.
- [x] Run `make check` and confirm all tests pass.

## Result

`make eval-retrieval`, 33 questions (negative tier excluded), gold symbols hand-labeled from source.

| configuration | recall@5 | recall@10 | MRR | nDCG@5 |
|---|---|---|---|---|
| BM25 only | 39.4% | 54.5% | 0.293 | 0.300 |
| Dense only | 69.7% | 81.8% | 0.419 | 0.472 |
| RRF hybrid | 63.6% | 81.8% | 0.439 | 0.468 |
| RRF + symbol pre-check | 97.0% | 97.0% | 0.924 | 0.929 |

> **Superseded by task 6.4.** Both findings below were measured on the dev set, where 31 of 33 questions name their own gold symbol. The held-out set reverses the first and withdraws the second. Kept here as written, because the correction is the point: see the *Held-out correction* section at the end.

Both headline findings contradict what `README.md` claimed before the measurement.

1. **RRF hybrid is worse than dense alone at k=5** (63.6% against 69.7%), and ties it at k=10. BM25 ranks the correct chunk badly enough that fusing it in costs top-of-list accuracy. RRF does raise MRR slightly, so it orders its hits better while finding fewer of them.
2. **The symbol pre-check carries the retriever**, from 63.6% to 97.0%. It was documented as a heuristic detail.

**Confound, reported in the results file and the README.** 31 of 33 questions name their gold symbol verbatim, and the pre-check is exact name lookup. On the 2 questions that do not name it, every configuration scores 0/2 or 1/2, and the pre-check's single miss is one of those two. The number is an upper bound for symbol-naming queries, not a general recall figure. Task 6.4 supplies the questions that test the rest.

**Answered from `open-questions.md`:** B1 keep the pre-check, B2 keep the reranker out of scope (no recall@5 to recall@10 gap remains once the pre-check is in).

**New finding, filed as C7:** `RunnableMap = RunnableParallel` at `runnables/base.py:4151` is a module-level assignment. The chunker skips all 265 of them, so that line is not in the index and q21 had to be graded against `RunnableParallel`.

**New option, filed under B1:** dense + pre-check without BM25 is untested and would delete the BM25 index and its 380 ms cold start.

## Held-out correction (task 6.4)

The held-out set has 7 of 15 graded questions naming their symbol, against the dev set's 31 of 33.

| configuration | dev recall@5 | held-out recall@5 |
|---|---|---|
| BM25 only | 39.4% | 73.3% |
| Dense only | 69.7% | 86.7% |
| RRF hybrid | 63.6% | **93.3%** |
| RRF + symbol pre-check | **97.0%** | 93.3% |

1. **"RRF hybrid is worse than dense alone" does not survive.** It was the best configuration on held-out. BM25 nearly doubles once questions stop naming the answer, which is the workload a lexical index is for. The 6.3 proposal to delete BM25 is withdrawn.
2. **"The symbol pre-check carries the retriever" does not survive either.** On held-out it adds zero recall over plain RRF, and scores identically inside both halves of the phrasing split (7/7 and 7/8). It still improves ranking: MRR 0.880 against 0.813. The 97% measured symbol lookup, not retrieval.
3. **The reranker conclusion holds.** No recall@5 to recall@10 gap on either set for either top configuration.

The lesson is about the harness, not the retriever: a 33-question set where 94% of questions name their own answer cannot rank retrieval strategies. Both sets remain too small to settle the architecture, since one question moves held-out recall by 6.7 points.
