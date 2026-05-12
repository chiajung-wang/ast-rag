.PHONY: install run index eval check

install:
	uv sync || pip install -e ".[dev]"

index:
	python -m indexer

run:
	streamlit run app.py

eval:
	python evals/run.py

check:
	python -m pytest tests/ -v
