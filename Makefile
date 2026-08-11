.PHONY: install run index eval eval-test eval-retrieval check

install:
	uv sync || pip install -e ".[dev]"

index:
	python -m indexer

run:
	streamlit run app.py

eval:
	python evals/run.py

eval-test:
	python evals/run.py --questions evals/questions-test.jsonl

eval-retrieval:
	python -m evals.retrieval_eval
	python -m evals.retrieval_eval --questions evals/questions-test.jsonl

check:
	python -m pytest tests/ -v
