# Prefer `uv run` so targets work without activating .venv first.
# Falls back to bare python for a pip install.
PY := $(shell command -v uv >/dev/null 2>&1 && echo 'uv run python' || echo 'python')

.PHONY: install run index eval eval-test eval-retrieval check

install:
	uv sync || pip install -e ".[dev]"

index:
	$(PY) -m indexer

run:
	$(shell command -v uv >/dev/null 2>&1 && echo 'uv run' || echo '') streamlit run app.py

eval:
	$(PY) evals/run.py

eval-test:
	$(PY) evals/run.py --questions evals/questions-test.jsonl

eval-retrieval:
	$(PY) -m evals.retrieval_eval
	$(PY) -m evals.retrieval_eval --questions evals/questions-test.jsonl

check:
	$(PY) -m pytest tests/ -v
