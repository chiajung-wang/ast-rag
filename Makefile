.PHONY: install run index eval eval-retrieval check

install:
	uv sync || pip install -e ".[dev]"

index:
	python -m indexer

run:
	streamlit run app.py

eval:
	python evals/run.py

eval-retrieval:
	python -m evals.retrieval_eval

check:
	python -m pytest tests/ -v
