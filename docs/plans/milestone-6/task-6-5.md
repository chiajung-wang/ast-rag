# Task 6.5 — Inheritance-Aware Class Outline

## Goal

Give `get_class_outline` the inheritance information it lacks, then delete the langchain-core naming rules from the system prompt. The prompt hack exists because the tool is incomplete. Fix the tool and the hack becomes unnecessary.

## Acceptance Criteria

- [ ] Each class chunk records its base class names.
- [ ] `get_class_outline` returns the methods of the class and the methods of its base classes inside the corpus.
- [ ] The output marks each method with the class that defines it.
- [ ] `agent/answer_node.py` no longer names any langchain-core class.
- [ ] The held-out test score from task 6.4 does not drop after the prompt hack is removed.
- [ ] `make index` rebuilds the index with the new column.
- [ ] `make check` passes.

## Problem

`storage/db.py:129-141` selects rows where `parent_class` matches the requested name. It returns the methods that the class defines itself. It does not return inherited methods, and it does not know that a class has a base class at all.

The system prompt fills the gap with hardcoded rules:

```
"Async sibling: for Base* classes drop the 'Base' prefix ... ALWAYS call
get_class_outline on the async sibling"
"All mixin/parent classes listed in the class definition"
```

The second rule asks the model to read base classes out of the source text that the chunk happens to contain. The first rule is a naming convention that holds inside `langchain-core` and nowhere else.

A second defect sits in the same query. `class_outline` matches on name across every file, so two classes with the same name merge into one outline. The current index holds 5 such pairs: `NoLock`, `RunInfo`, `Tee`, `ToolCall`, `ToolCallChunk`.

## Design

### Index time: capture base classes

`indexer/chunker.py:48` visits each `ast.ClassDef`. The node carries `node.bases`. Extract the simple names.

```python
def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            names.append(b.id)
        elif isinstance(b, ast.Attribute):
            names.append(b.attr)
    return names
```

Skip subscripted bases such as `Generic[T]`, or keep the outer name only. Store the result as a JSON list in a new `base_classes` column on `chunks`, and add the field to `Chunk`.

The column changes the schema. `_init_schema` uses `CREATE TABLE IF NOT EXISTS`, so an existing `index.db` will not gain the column. State in `README.md` that milestone 6 needs one `make index` rebuild, and add an `ALTER TABLE` guard or a schema-version check.

The chunk hash covers `text`, so unchanged chunks keep their embeddings. A rebuild costs almost nothing in OpenAI calls.

### Query time: walk the MRO

`class_outline(name)` becomes:

1. Find the class chunk. Apply the same tie-break rule that task 6.1 adds to `symbol_lookup`, so two same-named classes no longer merge.
2. Collect its methods.
3. For each base class name, look up that class in the corpus and collect its methods. Recurse, with a depth cap of 3 and a visited set to stop cycles.
4. A subclass method overrides a base method with the same name. Keep the subclass one, and mark the base one as overridden or drop it.
5. Return the rows in the order class first, then base classes.

Bases outside the corpus (`BaseModel`, `ABC`, `Generic`) will not resolve. That is correct. Note them in the output as external, so the model knows the outline is partial.

Output format keeps the current shape and adds the defining class:

```
[callbacks/base.py:120-124] BaseCallbackHandler.on_llm_start(...)   # Run when LLM starts running
[callbacks/base.py:301-305] (inherited from LLMManagerMixin) on_llm_end(...)
[external] BaseModel — not in corpus
```

### Prompt: delete the hack

Remove STEP 1b entirely. Replace STEP 1 with a general instruction:

```
STEP 1 — Call get_class_outline on the relevant class first. It returns every method
of that class and of its base classes inside the corpus, in one call. For a standalone
function, call read_file directly.
```

Keep STEP 1c, STEP 1d, and STEP 2. Those are general instructions, not corpus lore.

### Verify

Run the held-out test set from task 6.4 before and after the prompt change. If the score drops, the outline is missing something that the hack supplied. Find that gap in the tool. Do not put the class names back.

## Files

- `storage/chunk.py` — add `base_classes: list[str]`
- `indexer/chunker.py` — extract base names
- `storage/db.py` — `base_classes` column, MRO walk in `class_outline`, schema-version guard
- `agent/answer_node.py` — new `get_class_outline` docstring, delete STEP 1b
- `tests/test_chunker.py`, `tests/test_db.py`, `tests/test_answer_node.py` — new tests
- `README.md`, `CLAUDE.md`, `CONTEXT.md` — describe the inheritance-aware outline

## Steps

- [ ] Add `base_classes` to `Chunk` and to `make_chunk`.
- [ ] Extract base names in `chunk_file` and add a test with a multi-base class.
- [ ] Add the `base_classes` column and a schema-version guard to `storage/db.py`.
- [ ] Rewrite `class_outline` as an MRO walk with a depth cap of 3 and a visited set.
- [ ] Add a test: a subclass outline includes the base class methods, marked with the defining class.
- [ ] Add a test: two classes with the same name in different files do not merge.
- [ ] Add a test: an override hides the base method of the same name.
- [ ] Update the `get_class_outline` tool docstring.
- [ ] Delete STEP 1b from the system prompt and rewrite STEP 1.
- [ ] Run `make index` and confirm the chunk count is unchanged.
- [ ] Run the held-out test set before and after. Record both scores here.
- [ ] Update `README.md`, `CLAUDE.md`, and `CONTEXT.md`.
- [ ] Run `make check` and confirm all tests pass.
