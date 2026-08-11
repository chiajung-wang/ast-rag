# Task 6.3 — Retrieval Metrics and Ablation

## Goal

Measure the retriever on its own. Today the eval scores answers end to end, so a failure could come from retrieval or from generation, and nothing separates them. Then prove that hybrid retrieval plus RRF beats each single leg.

This task produces the strongest artifact in the milestone. It converts the architecture from an asserted design into a measured result.

## Acceptance Criteria

- [ ] Every non-negative question in `evals/questions.jsonl` carries an `expected_symbols` list.
- [ ] `evals/retrieval_eval.py` reports recall@5, recall@10, MRR, and nDCG@5 over the question set.
- [ ] The retrieval eval runs with no LLM call and finishes in under 60 seconds.
- [ ] `make eval-retrieval` runs it.
- [ ] The ablation table compares 4 configurations and lands in `evals/results/`.
- [ ] `README.md` shows the ablation table.
- [ ] `make check` passes.

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

- [ ] Add `expected_symbols` to every non-negative question in `evals/questions.jsonl`.
- [ ] Write the metric functions in `evals/retrieval_eval.py`.
- [ ] Add unit tests for recall@k, MRR, and nDCG@5 against hand-built hit lists with known answers.
- [ ] Add the embedding cache, keyed by the question text.
- [ ] Write `run_config` and wire the 4 retriever configurations.
- [ ] Write `format_ablation_md` and the `__main__` entry point.
- [ ] Add `eval-retrieval` to the `Makefile`.
- [ ] Run the ablation and write the result to `evals/results/`.
- [ ] Copy the table into `README.md` with one sentence that states which configuration wins.
- [ ] Run `make check` and confirm all tests pass.
