# Task 6.2 — Honest Claims

## Goal

Make every statement in `README.md` and `CONTEXT.md` true against the code. A reader who opens the named file must find the described behavior.

## Acceptance Criteria

- [x] `README.md` no longer says that the two searches run in parallel, or the code runs them in parallel.
- [x] `README.md` describes what citation validation proves and what it does not prove.
- [x] `CONTEXT.md` matches `agent/citations.py` on prefix stripping, or the code strips the prefix.
- [x] The eval baseline in `README.md` states the run count, and states that a single run carries no variance figure.
- [x] Every `build_permalink` URL resolves at the pinned commit SHA.
- [x] A citation written with a corpus-root prefix survives validation and reaches the reader as a canonical short path.
- [x] `make check` passes.

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
- **Option B:** make it true. Run the BM25 search in a thread while the embedding call is in flight, then join.

**Resolved: Option A.** This task first recommended Option B. Measurement reversed the decision. BM25 search over the 2414-chunk index costs 0.6 ms to 1.1 ms per query. The OpenAI embedding call costs about 300 ms. A thread would overlap under 1% of the wall-clock time, and it would add a thread pool to maintain. Fix the sentence instead.

The measurement did find a real cost. `BM25Index.from_db` takes 379 ms, and it runs at the first query of each process. That is 400 times the cost of one search. It is a cold-start problem, not a per-query problem. Record it under **Limitations** in `README.md`.

### 2. Citation validation is weaker than the word "validated" implies — `storage/db.py:105-113`

`chunk_exists_at` uses `line_start <= ? AND line_end >= ?`. That is containment. A citation of `[runnables/base.py:900-905]` passes because the class chunk at 891-1205 contains that range. The check proves that the range sits inside an indexed symbol. It does not prove that the range supports the claim.

Fix: state the guarantee exactly, in `README.md` and in `CONTEXT.md` under **Citation Marker**.

> Validation confirms that the cited line range falls inside an indexed symbol in that file. It does not confirm that those lines support the claim.

Task 6.7 adds the metric that measures the stronger property.

### 3. Prefix stripping: `CONTEXT.md` says one thing, the code does another

`CONTEXT.md:18` states: "One `CORPUS_ROOT` constant handles prefix stripping at validation time." `validate_citations` in `agent/citations.py` passes the raw path straight to `chunk_exists_at`, which matches `file_path` exactly. A model that emits `langchain_core/runnables/base.py:891-1205` writes a correct citation, and the validator removes it.

Fix the code, because the documented behavior is the better behavior. Strip a leading `langchain_core/` or a leading corpus-root prefix before the lookup. Add a test for a citation that carries the prefix.

### 4. Every GitHub permalink is broken — `ui/helpers.py:12-16`

Found while checking claim 2. `build_permalink` hardcodes `libs/core/` in the URL:

```python
f"https://github.com/langchain-ai/langchain/blob/{COMMIT_SHA}/libs/core/{path}#L{line_start}-L{line_end}"
```

Chunk `file_path` is relative to `CORPUS_ROOT`, which is `langchain/libs/core/langchain_core`. So `file_path` is `runnables/base.py`, and the URL omits the `langchain_core/` segment. Verified against GitHub at the pinned SHA:

| URL path | Status |
|---|---|
| `libs/core/runnables/base.py` | 404 |
| `libs/core/langchain_core/runnables/base.py` | 200 |

Every "View on GitHub" link in the UI is dead. `README.md:3` sells "citations that link directly to the relevant lines on GitHub".

Fix: derive the prefix from `CORPUS_SUBPATH` and `REPO_URL` in `indexer/corpus_config.py` instead of hardcoding it. The two cannot drift apart then. Correct the URL template in `CONTEXT.md` under **Citation Expander**.

### 5. The baseline reports a single run — `README.md:135`

`evals/run.py` defaults to `n_runs=3` and reports a median and a variance. The README publishes an n=1 number. Report the run count and the variance, or re-run at n=3 and publish that.

## Files

- `agent/citations.py` — `normalize_path`, canonical rewrite of surviving markers
- `ui/helpers.py` — permalink prefix from `CORPUS_SUBPATH`
- `README.md` — search claim, validation guarantee, baseline caveats, Limitations
- `CONTEXT.md` — validation guarantee, corrected permalink template
- `tests/test_citations.py`, `tests/test_ui_helpers.py` — new tests

## Steps

- [x] Decide between Option A and Option B for the parallel claim. Measured first: BM25 costs ~1 ms against a ~300 ms embedding call, so Option A.
- [x] Rewrite the search description in `README.md` with both measured figures.
- [x] Add `normalize_path` to `agent/citations.py` and derive the prefixes from `CLONE_DIR` and `CORPUS_SUBPATH`.
- [x] Rewrite surviving markers to the canonical short path, so downstream consumers see one form.
- [x] Fix `build_permalink` to take its prefix from `CORPUS_SUBPATH`.
- [x] Rewrite the citation guarantee in `README.md` and `CONTEXT.md`.
- [x] Correct the permalink template in `CONTEXT.md`.
- [x] Add a `Limitations` section to `README.md` (cold start, single session, name collisions, containment check).
- [x] Annotate the baseline with n=1, prompt-tuning contamination, and the tier counts.
- [ ] Re-run `make eval` at n=3. **Deferred — needs the user's API budget.** The README states n=1 in the meantime.
- [x] Run `make check` and confirm all tests pass.

## Verification

- New tests fail against the old code: both permalink tests, and `chunk_exists_at("langchain_core/runnables/base.py", 10, 100)` returns `False` before normalization.
- The corrected permalink returns HTTP 200 against GitHub at the pinned SHA. The old one returns 404.
- Tests: 96 to 109.
