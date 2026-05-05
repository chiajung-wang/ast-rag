# Task 1.1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `Makefile`
- Create: `conftest.py`
- Create: `storage/__init__.py`
- Create: `indexer/__init__.py`
- Create: `retrieval/__init__.py`
- Create: `agent/__init__.py`
- Create: `evals/__init__.py`
- Create: `tests/__init__.py`

---

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "ast-rag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "openai>=1.0",
    "rank-bm25>=0.2",
    "sqlite-vec>=0.1",
    "streamlit>=1.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `Makefile`**

```makefile
.PHONY: install run index eval

install:
	uv sync || pip install -e ".[dev]"

index:
	python -m indexer

run:
	streamlit run app.py

eval:
	python evals/run.py
```

- [ ] **Step 3: Create package `__init__.py` files**

Run:
```bash
mkdir -p storage indexer retrieval agent evals tests
touch storage/__init__.py indexer/__init__.py retrieval/__init__.py agent/__init__.py evals/__init__.py tests/__init__.py conftest.py
```

- [ ] **Step 4: Verify install works**

Run:
```bash
make install
```

Expected: dependencies install without error. `python -c "import openai, rank_bm25, sqlite_vec, streamlit, langgraph"` prints nothing (no ImportError).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml Makefile conftest.py storage/ indexer/ retrieval/ agent/ evals/ tests/
git commit -m "chore: project scaffold — packages, Makefile, pyproject.toml"
```
