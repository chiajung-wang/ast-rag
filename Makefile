.PHONY: install run index eval

install:
	uv sync || pip install -e ".[dev]"

index:
	python -m indexer

run:
	streamlit run app.py

eval:
	python evals/run.py
