# Task 6.2 — Honest Claims

## Goal

Make every statement in `README.md` and `CONTEXT.md` true against the code. A reader who opens the named file must find the described behavior.

## Acceptance Criteria

- [ ] `README.md` no longer says that the two searches run in parallel, or the code runs them in parallel.
- [ ] `README.md` describes what citation validation proves and what it does not prove.
- [ ] `CONTEXT.md` matches `agent/citations.py` on prefix stripping, or the code strips the prefix.
- [ ] The eval baseline in `README.md` reports the number of runs and the variance.
- [ ] `make check` passes.

## Claims to fix

### 1. "Two searches in parallel" — `README.md:30` vs `retrieval/pipeline.py:54-58`

```python
bm25_results = _get_bm25().search(query, k=10)
embedding = _embed_query(query)          # blocking network call
dense_results = _get_db().vector_search(embedding, k=10)
```

The searches run in sequence. The OpenAI embedding call blocks between them.

Pick one:

- **Option A (cheap):** change the README to "Each query runs two searches." Remove "in parallel".
- **Option B (better):** make it true. Run the BM25 search in a thread while the embedding call is in flight, then join. BM25 over 2414 chunks is CPU work, so a thread gives a real overlap with the network wait.

Option B is preferred. It is about 10 lines, and it makes the sentence a result instead of a description.

### 2. Citation validation is weaker than the word "validated" implies — `storage/db.py:105-113`

`chunk_exists_at` uses `line_start <= ? AND line_end >= ?`. That is containment. A citation of `[runnables/base.py:900-905]` passes because the class chunk at 891-1205 contains that range. The check proves that the range sits inside an indexed symbol. It does not prove that the range supports the claim.

Fix: state the guarantee exactly, in `README.md` and in `CONTEXT.md` under **Citation Marker**.

> Validation confirms that the cited line range falls inside an indexed symbol in that file. It does not confirm that those lines support the claim.

Task 6.7 adds the metric that measures the stronger property.

### 3. Prefix stripping: `CONTEXT.md` says one thing, the code does another

`CONTEXT.md:18` states: "One `CORPUS_ROOT` constant handles prefix stripping at validation time." `validate_citations` in `agent/citations.py` passes the raw path straight to `chunk_exists_at`, which matches `file_path` exactly. A model that emits `langchain_core/runnables/base.py:891-1205` writes a correct citation, and the validator removes it.

Fix the code, because the documented behavior is the better behavior. Strip a leading `langchain_core/` or a leading corpus-root prefix before the lookup. Add a test for a citation that carries the prefix.

### 4. The baseline reports a single run — `README.md:135`

`evals/run.py` defaults to `n_runs=3` and reports a median and a variance. The README publishes an n=1 number. Report the run count and the variance, or re-run at n=3 and publish that.

## Files

- `retrieval/pipeline.py` — optional thread for the BM25 search (Option B)
- `agent/citations.py` — strip the corpus-root prefix before validation
- `README.md` — parallel claim, validation guarantee, baseline with variance
- `CONTEXT.md` — validation guarantee
- `tests/test_citations.py` — prefixed-path test
- `tests/test_pipeline.py` — test for the threaded search, if Option B

## Steps

- [ ] Decide between Option A and Option B for the parallel claim. Option B is preferred.
- [ ] If Option B: run the BM25 search in a `ThreadPoolExecutor` while `_embed_query` runs, then join.
- [ ] Add prefix stripping to `validate_citations` and a test for a prefixed path.
- [ ] Rewrite the citation guarantee in `README.md` and `CONTEXT.md`.
- [ ] Re-run `make eval` at n=3, or annotate the README number with "n=1, no variance measured".
- [ ] Run `make check` and confirm all tests pass.
