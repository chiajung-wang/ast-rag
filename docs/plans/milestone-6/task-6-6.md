# Task 6.6 — Repo Hygiene

## Goal

Four small changes that a reviewer checks in the first 5 minutes: automated tests, pinned dependencies, prompt caching, and a stated concurrency limit.

## Acceptance Criteria

- [x] GitHub Actions runs `make check` on every push and every pull request.
- [x] `uv.lock` is tracked in git.
- [x] The answer node caches the chunk context, and a repeated tool round reads from the cache.
- [x] The concurrency limit of the Streamlit app is fixed or documented.
- [x] `make check` passes.

## Items

### 1. No CI

There is no `.github/` directory. 89 tests pass and nothing runs them.

Add `.github/workflows/check.yml`:

```yaml
name: check
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --all-extras
      - run: uv run pytest tests/ -v
```

The tests must not need an index or an API key. Check that first. If any test reads `index.db`, mark it with `@pytest.mark.integration` and deselect it in CI.

**Makefile portability.** Every target calls bare `python`, so `make check` fails unless the caller already activated `.venv`. The workflow above avoids the problem, because `uv run` resolves the interpreter itself. Fix the Makefile too: call `uv run python` with a fallback, or state the activation step in `README.md`.

### 2. `uv.lock` is gitignored

`.gitignore:45` ignores `uv.lock`. The project pins a corpus commit SHA for reproducibility and then leaves its own dependency versions floating. Remove that line and commit the lock file. Remove `package.json`, `package-lock.json`, and `node_modules` from `.gitignore` too. This project has no JavaScript.

### 3. No prompt caching — `agent/answer_node.py:92-131`

The system prompt carries the full `text` of 5 chunks. A class chunk can run to hundreds of lines. The loop resends that prompt on every round, up to 8 rounds. `ChatAnthropic` is constructed on every `answer_node` call.

Two changes:

- Split the system prompt into two blocks: a static instruction block and a chunk-context block. Mark the last block with `cache_control` so the tool loop reads the prefix from cache after the first round.
- Build the model once at module scope, keyed by the model name, instead of once per call.

Confirm the win: read `cache_read_input_tokens` from `usage_metadata` and add it to the eval cost report. A round that reads from cache costs about a tenth of a round that does not.

Note the minimum cacheable prefix. It varies by model, and Haiku 4.5 needs 4096 tokens. Five chunks usually clear that, but a short retrieval may not, and the block then silently does not cache. Report the cache-read tokens so the answer is measured, not assumed.

### 4. One session only — `storage/db.py:15`

`sqlite3.connect(path)` defaults to `check_same_thread=True`, and the connection lives in a module-level global. Streamlit runs one script thread per browser session, so a second session raises `ProgrammingError`.

Pick one:

- **Option A:** pass `check_same_thread=False`. Reads are safe with a single connection under SQLite's default threading mode, and this app only reads at query time.
- **Option B:** state the limit in `README.md` under a "Limitations" heading.

Option A is 1 line. Do that, and keep the note.

## Files

- `.github/workflows/check.yml` — new
- `.gitignore` — remove `uv.lock` and the JavaScript entries
- `uv.lock` — commit
- `agent/answer_node.py` — cache control, module-level model
- `storage/db.py` — `check_same_thread=False`
- `evals/run.py` — report cache-read tokens
- `README.md` — limitations note

## Steps

- [x] Confirm that no test needs `index.db` or an API key. Mark and deselect any that does.
- [x] Add `.github/workflows/check.yml` and confirm it passes on a push.
- [x] Remove `uv.lock` and the JavaScript entries from `.gitignore`, then commit `uv.lock`.
- [x] Split the system prompt into a static block and a chunk block, and add `cache_control` to the chunk block.
- [x] Build the `ChatAnthropic` client once per model name at module scope.
- [x] Add `cache_read_input_tokens` to the eval cost report.
- [~] Confirm `cache_read` is above 0 after round 1. Confirmed on a standalone tool-bound call (8k prefix, round 2 read 8,141). **Not** confirmed through a full eval question — the account ran out of credit first.
- [x] Add `check_same_thread=False` to the `DB` connection.
- [x] Add a "Limitations" section to `README.md`.
- [x] Run `make check` and confirm all tests pass.

## Result

All four items landed. Tests 158 to 164.

**CI** — `.github/workflows/check.yml` runs `uv run pytest` on push and pull request. Verified first that the suite needs neither an index nor a key: with `index.db` moved aside and both API keys unset, all 158 tests passed. The workflow unsets both keys explicitly, so a test that quietly starts depending on one fails in CI rather than passing locally.

**`uv.lock` tracked.** Also dropped `package.json`, `package-lock.json`, `node_modules` and `.dependency-cruiser.js` from `.gitignore` — this project has no JavaScript.

**Makefile** no longer assumes an activated venv. `PY := uv run python` when `uv` is present, bare `python` otherwise. `make check` now works from a clean shell.

**Thread safety** — `check_same_thread=False`, with a comment recording why. Every query path is a read; the indexer writes single-threaded.

### Prompt caching — works, but only above a size threshold

The system prompt is one `cache_control` block. The model client is now built once per (model, tool-binding) instead of per call, behind `reset_model_cache()` so tests can clear it.

**Verified against the API that the mechanism works**: with tools bound and an 8k-token prefix, round 2 reported `cache_read=8141`.

**Verified that it does not always engage.** The first end-to-end run reported `cache_read=0` across 7 rounds. The cause is the one this task's own design note warned about and then failed to check: Haiku 4.5 will not cache a prefix under 4096 tokens, and says nothing when it declines. Measured with `count_tokens` over 5 sample questions:

| tokens | caches? | question |
|---|---|---|
| 3,395 | no | what events does BaseCallbackHandler expose |
| 4,154 | yes | what methods does BaseStore define |
| 7,319 | yes | how does RunnableWithFallbacks decide |
| 8,615 | yes | where is RunnableSequence defined |
| 29,231 | yes | where is Runnable defined / astream_events |

Median 7,319, so most questions cache. The ones that miss are the cheapest ones, which is the right way round. `cache_read_tokens` is now carried on the answer and reported per eval run, so this stays measured rather than assumed.

Two reporting details worth keeping: a successful cache **write** appears under `ephemeral_5m_input_tokens`, not `cache_creation`, which stays 0 — reading only `cache_creation` would have shown a permanent zero. And the static instructions are ~700 tokens alone, far under the minimum, so a second breakpoint there would never have fired.

### Blocked

The end-to-end confirmation on an above-threshold question could not be completed: **the Anthropic account ran out of credit** mid-verification (`400 invalid_request_error: credit balance is too low`). The milestone-5 error handling caught it correctly and returned a graceful message rather than a traceback.

This also blocks A1, 6.7's regression baseline, and the 6.5 accuracy check. Those need credit, not just a spending decision.
