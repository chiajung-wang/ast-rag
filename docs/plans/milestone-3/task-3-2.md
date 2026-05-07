# Task 3.2: Citation Validator

**What to build:** Pure function that parses `[path:start-end]` citation markers from answer text, validates each via `db.chunk_exists_at`, strips invalid ones, and appends a footnote when any are removed.

**Blocked by:** Task 1.4 (DB layer — `chunk_exists_at` already exists)

**Acceptance criteria:**
- [ ] Valid citations pass through unchanged
- [ ] Invalid citation stripped from text; footnote `"*N citation(s) could not be verified and were removed.*"` appended
- [ ] Multiple invalid citations: all stripped, count in footnote is correct
- [ ] Text with no citations returns unchanged
- [ ] Mix of valid + invalid: valid kept, invalid stripped
- [ ] All tests pass

---

**Files:**
- Create: `agent/citations.py`
- Create: `tests/test_citations.py`

---

- [ ] **Step 1: Write the failing tests**

`tests/test_citations.py`:

```python
import pytest
from storage.chunk import make_chunk
from storage.db import DB
from agent.citations import validate_citations


@pytest.fixture
def db(tmp_path):
    d = DB(str(tmp_path / "test.db"))
    chunk = make_chunk("runnables/base.py", "RunnableSequence", "class", None, 10, 100, None, "class RS: pass")
    d.insert_chunk(chunk)
    return d


def test_valid_citation_unchanged(db):
    text = "Defined in [runnables/base.py:10-100]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:10-100]" in result
    assert "could not be verified" not in result


def test_invalid_citation_stripped(db):
    text = "See [runnables/base.py:999-1000] for details."
    result = validate_citations(text, db)
    assert "[runnables/base.py:999-1000]" not in result
    assert "*1 citation(s) could not be verified and were removed.*" in result


def test_multiple_invalid_citations_stripped(db):
    text = "See [runnables/base.py:999-1000] and [other/file.py:1-5]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:999-1000]" not in result
    assert "[other/file.py:1-5]" not in result
    assert "*2 citation(s) could not be verified and were removed.*" in result


def test_no_citations_unchanged(db):
    text = "This answer has no citations."
    result = validate_citations(text, db)
    assert result == text


def test_mixed_valid_and_invalid(db):
    text = "Valid [runnables/base.py:10-100] and invalid [fake/path.py:1-2]."
    result = validate_citations(text, db)
    assert "[runnables/base.py:10-100]" in result
    assert "[fake/path.py:1-2]" not in result
    assert "*1 citation(s) could not be verified and were removed.*" in result
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
.venv\Scripts\python -m pytest tests/test_citations.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.citations'`

- [ ] **Step 3: Write `agent/citations.py`**

```python
from __future__ import annotations
import re
from storage.db import DB

_CITATION_RE = re.compile(r'\[([^:\]\s]+):(\d+)-(\d+)\]')


def validate_citations(text: str, db: DB) -> str:
    matches = _CITATION_RE.findall(text)
    if not matches:
        return text

    invalid: list[str] = []
    for path, start, end in matches:
        if not db.chunk_exists_at(path, int(start), int(end)):
            invalid.append(f"[{path}:{start}-{end}]")

    if not invalid:
        return text

    result = text
    for marker in invalid:
        result = result.replace(marker, "")

    n = len(invalid)
    result = result.rstrip() + f"\n\n*{n} citation(s) could not be verified and were removed.*"
    return result
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv\Scripts\python -m pytest tests/test_citations.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add agent/citations.py tests/test_citations.py
git commit -m "feat(agent): citation validator — parse, verify, strip, footnote"
```
