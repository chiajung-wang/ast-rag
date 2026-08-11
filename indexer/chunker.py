from __future__ import annotations
import ast
from pathlib import Path
from storage.chunk import Chunk, make_chunk


def _get_docstring(node: ast.AST) -> str | None:
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value
    return None


def _extract_source(lines: list[str], node: ast.AST) -> str:
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def _base_names(node: ast.ClassDef) -> list[str]:
    """Simple names of a class's bases.

    `Foo` yields "Foo", `mod.Foo` yields "Foo", and `Generic[T]` yields
    "Generic". Anything else (a call, a comprehension) is skipped. Names are
    resolved against the corpus later, so an unresolvable base is harmless.
    """
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def chunk_file(file_path: Path, corpus_root: Path) -> list[Chunk]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    lines = source.splitlines()
    rel_path = str(file_path.relative_to(corpus_root)).replace("\\", "/")
    chunks: list[Chunk] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                make_chunk(
                    file_path=rel_path,
                    symbol_name=node.name,
                    symbol_type="function",
                    parent_class=None,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=_get_docstring(node),
                    text=_extract_source(lines, node),
                )
            )
        elif isinstance(node, ast.ClassDef):
            chunks.append(
                make_chunk(
                    file_path=rel_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    parent_class=None,
                    line_start=node.lineno,
                    line_end=node.end_lineno,
                    docstring=_get_docstring(node),
                    text=_extract_source(lines, node),
                    base_classes=_base_names(node),
                )
            )
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        make_chunk(
                            file_path=rel_path,
                            symbol_name=item.name,
                            symbol_type="method",
                            parent_class=node.name,
                            line_start=item.lineno,
                            line_end=item.end_lineno,
                            docstring=_get_docstring(item),
                            text=_extract_source(lines, item),
                        )
                    )

    return chunks


def chunk_corpus(corpus_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for py_file in sorted(corpus_root.rglob("*.py")):
        chunks.extend(chunk_file(py_file, corpus_root))
    return chunks
