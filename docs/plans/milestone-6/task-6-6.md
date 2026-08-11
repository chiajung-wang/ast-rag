# Task 6.6 — Repo Hygiene

## Goal

Four small changes that a reviewer checks in the first 5 minutes: automated tests, pinned dependencies, prompt caching, and a stated concurrency limit.

## Acceptance Criteria

- [ ] GitHub Actions runs `make check` on every push and every pull request.
- [ ] `uv.lock` is tracked in git.
- [ ] The answer node caches the chunk context, and a repeated tool round reads from the cache.
- [ ] The concurrency limit of the Streamlit app is fixed or documented.
- [ ] `make check` passes.

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

- [ ] Confirm that no test needs `index.db` or an API key. Mark and deselect any that does.
- [ ] Add `.github/workflows/check.yml` and confirm it passes on a push.
- [ ] Remove `uv.lock` and the JavaScript entries from `.gitignore`, then commit `uv.lock`.
- [ ] Split the system prompt into a static block and a chunk block, and add `cache_control` to the chunk block.
- [ ] Build the `ChatAnthropic` client once per model name at module scope.
- [ ] Add `cache_read_input_tokens` to the eval cost report.
- [ ] Run one eval question and confirm that `cache_read_input_tokens` is above 0 after round 1.
- [ ] Add `check_same_thread=False` to the `DB` connection.
- [ ] Add a "Limitations" section to `README.md`.
- [ ] Run `make check` and confirm all tests pass.
