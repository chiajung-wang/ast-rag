from __future__ import annotations
import json
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from storage.chunk import Chunk, make_chunk

EMBEDDING_DIM = 1536
MAX_MRO_DEPTH = 3
EMBED_MODEL_KEY = "embed_model"


class EmbedModelMismatch(RuntimeError):
    """Raised when the index and the configured embedding model disagree."""


@dataclass
class OutlineEntry:
    """One method in a class outline, tagged with the class that defines it."""
    chunk: Chunk
    defining_class: str
    inherited: bool


@dataclass
class ClassOutline:
    class_chunk: Chunk | None
    entries: list[OutlineEntry] = field(default_factory=list)
    external_bases: list[str] = field(default_factory=list)   # bases outside the corpus
    subclasses: list[Chunk] = field(default_factory=list)     # direct subclasses in corpus

    def __bool__(self) -> bool:
        return self.class_chunk is not None


def _serialize(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _deserialize(b: bytes) -> list[float]:
    return list(struct.unpack(f"{len(b) // 4}f", b))


class DB:
    def __init__(self, path: str = "index.db"):
        # Streamlit runs one script thread per browser session, and this
        # connection lives in a module-level global, so a second session hit
        # "SQLite objects created in a thread can only be used in that same
        # thread".
        #
        # check_same_thread=False lifts Python's guard but does NOT make the
        # connection safe to share -- that depends on how SQLite was compiled.
        # sqlite3.threadsafety is 3 (serialized) on macOS here but 1
        # (multi-thread, connection not shareable) on the Linux CI runner,
        # where four threads reading concurrently raised InterfaceError and
        # IndexError. The flag only permits sharing; _lock makes it safe.
        # Reentrant, so a locked method may call another locked method.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    # ── connection access ─────────────────────────────────────────────────
    # Every statement goes through these, so the lock cannot be forgotten.
    # Rows are materialised while the lock is held: handing back a live cursor
    # would move the real connection access outside it.

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self.conn.execute(sql, params)

    def _executescript(self, script: str) -> None:
        with self._lock:
            self.conn.executescript(script)

    def _commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def _init_schema(self) -> None:
        self._executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                file_path   TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                parent_class TEXT,
                line_start  INTEGER NOT NULL,
                line_end    INTEGER NOT NULL,
                docstring   TEXT,
                text        TEXT NOT NULL,
                embed_text  TEXT NOT NULL,
                base_classes TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_symbol
                ON chunks(lower(symbol_name));
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding FLOAT[{EMBEDDING_DIM}]
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # CREATE TABLE IF NOT EXISTS will not add a column to an index built
        # before base_classes existed. Migrate in place so an old .db keeps
        # working; the column stays empty until the next `make index`.
        existing = {row[1] for row in self._fetchall("PRAGMA table_info(chunks)")}
        if "base_classes" not in existing:
            self._exec(
                "ALTER TABLE chunks ADD COLUMN base_classes TEXT NOT NULL DEFAULT '[]'"
            )
        self._commit()

    def insert_chunk(self, chunk: Chunk) -> int:
        """Insert a chunk, or refresh base_classes on one already stored.

        The chunk id excludes base_classes, so re-indexing an existing corpus
        fills the column in place without invalidating a single embedding.
        """
        # Write, commit and read-back are one critical section: another thread
        # must not interleave between the upsert and the rowid lookup.
        with self._lock:
            self._exec(
                """
                INSERT INTO chunks
                    (id, file_path, symbol_name, symbol_type, parent_class,
                     line_start, line_end, docstring, text, embed_text, base_classes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET base_classes = excluded.base_classes
                """,
                (
                    chunk.id, chunk.file_path, chunk.symbol_name, chunk.symbol_type,
                    chunk.parent_class, chunk.line_start, chunk.line_end,
                    chunk.docstring, chunk.text, chunk.embed_text,
                    json.dumps(chunk.base_classes),
                ),
            )
            self._commit()
            # Unconditional lookup: an upsert that took the UPDATE branch does
            # not give a reliable lastrowid.
            row = self._fetchone(
                "SELECT rowid FROM chunks WHERE id = ?", (chunk.id,)
            )
            return row["rowid"]

    def has_embedding(self, rowid: int) -> bool:
        return self._fetchone(
            "SELECT rowid FROM vec_chunks WHERE rowid = ?", (rowid,)
        ) is not None

    def insert_embedding(self, rowid: int, embedding: list[float]) -> None:
        self._exec(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize(embedding)),
        )
        self._commit()

    def get_embedding(self, rowid: int) -> list[float] | None:
        """Read a stored vector back. Used to check that a new embedding
        provider still lands in the same vector space as the index."""
        row = self._fetchone(
            "SELECT embedding FROM vec_chunks WHERE rowid = ?", (rowid,)
        )
        return _deserialize(row["embedding"]) if row else None

    def sample_embedded_chunks(self, limit: int) -> list[tuple[int, Chunk]]:
        """A spread of chunks that have embeddings, as (rowid, chunk)."""
        rows = self._fetchall(
            """
            SELECT c.rowid AS rowid, c.id, c.file_path, c.symbol_name, c.symbol_type,
                   c.parent_class, c.line_start, c.line_end, c.docstring, c.text,
                   c.embed_text, c.base_classes
            FROM chunks c JOIN vec_chunks v ON v.rowid = c.rowid
            ORDER BY c.rowid LIMIT ?
            """,
            (limit,),
        )
        return [(r["rowid"], _row_to_chunk(r)) for r in rows]

    def vector_search(self, embedding: list[float], k: int = 10) -> list[Chunk]:
        rows = self._fetchall(
            """
            SELECT c.id, c.file_path, c.symbol_name, c.symbol_type, c.parent_class,
                   c.line_start, c.line_end, c.docstring, c.text, c.embed_text, c.base_classes
            FROM vec_chunks v
            JOIN chunks c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (_serialize(embedding), k),
        )
        return [_row_to_chunk(r) for r in rows]

    def symbol_lookup(self, name: str) -> Chunk | None:
        row = self._fetchone(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text, base_classes
            FROM chunks WHERE lower(symbol_name) = lower(?)
            ORDER BY
                CASE symbol_type WHEN 'class' THEN 0 WHEN 'method' THEN 1 ELSE 2 END,
                file_path, line_start
            LIMIT 1
            """,
            (name,),
        )
        return _row_to_chunk(row) if row else None

    def chunk_exists_at(self, file_path: str, line_start: int, line_end: int) -> bool:
        return self._fetchone(
            """
            SELECT 1 FROM chunks
            WHERE file_path = ? AND line_start <= ? AND line_end >= ?
            LIMIT 1
            """,
            (file_path, line_start, line_end),
        ) is not None

    def all_chunks(self) -> list[Chunk]:
        rows = self._fetchall(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text, base_classes
            FROM chunks
            """
        )
        return [_row_to_chunk(r) for r in rows]

    def all_symbol_names(self) -> set[str]:
        rows = self._fetchall("SELECT symbol_name FROM chunks")
        return {r["symbol_name"] for r in rows}

    # ── embedding-model guard ─────────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        row = self._fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._exec(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._commit()

    def assert_embed_model(self, model: str) -> None:
        """Refuse to query an index built by a different embedding model.

        Two models do not share a vector space, so a mismatch does not raise
        anything on its own — it silently returns plausible, wrong neighbours.
        The index records its model, and this fails loudly instead.

        An index built before the column existed has no recorded model. That
        is allowed through, because the alternative is breaking every existing
        .db on upgrade.
        """
        indexed = self.get_meta(EMBED_MODEL_KEY)
        if indexed is None or indexed == model:
            return
        raise EmbedModelMismatch(
            f"Index was built with embedding model {indexed!r} but the current "
            f"EMBED_MODEL is {model!r}. Embeddings from different models are "
            f"not comparable. Run `make index` to rebuild, or set EMBED_MODEL "
            f"back to {indexed!r}."
        )

    def _find_class(self, class_name: str) -> Chunk | None:
        """Resolve a class by name. Ties break on file_path, then line_start."""
        row = self._fetchone(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text, base_classes
            FROM chunks
            WHERE lower(symbol_name) = lower(?) AND symbol_type = 'class'
            ORDER BY file_path, line_start
            LIMIT 1
            """,
            (class_name,),
        )
        return _row_to_chunk(row) if row else None

    def _methods_of(self, class_name: str, file_path: str) -> list[Chunk]:
        """Methods of one class. Scoped to its file so same-named classes
        in different modules do not merge (NoLock, RunInfo, Tee, ToolCall,
        ToolCallChunk each appear twice in this corpus)."""
        rows = self._fetchall(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text, base_classes
            FROM chunks
            WHERE lower(parent_class) = lower(?) AND file_path = ?
            ORDER BY line_start
            """,
            (class_name, file_path),
        )
        return [_row_to_chunk(r) for r in rows]

    def _subclasses_of(self, class_name: str) -> list[Chunk]:
        """Direct subclasses inside the corpus. 331 class chunks, so a scan
        is cheaper than teaching SQLite to read the JSON column."""
        rows = self._fetchall(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text, base_classes
            FROM chunks WHERE symbol_type = 'class'
            """
        )
        target = class_name.lower()
        return sorted(
            (c for c in map(_row_to_chunk, rows)
             if any(b.lower() == target for b in c.base_classes)),
            key=lambda c: (c.file_path, c.line_start),
        )

    def class_outline(self, class_name: str, max_depth: int = MAX_MRO_DEPTH) -> ClassOutline:
        """Methods of a class and of its base classes, walked breadth-first.

        Breadth-first matters: the first definition of a method name wins, so
        a subclass override hides the base method rather than the reverse.

        Bases outside the corpus (BaseModel, ABC, Generic) cannot resolve and
        are reported separately, so a caller knows the outline is partial.
        """
        root = self._find_class(class_name)
        if root is None:
            return ClassOutline(class_chunk=None)

        entries: list[OutlineEntry] = []
        external: list[str] = []
        seen_methods: set[str] = set()
        visited: set[str] = {root.symbol_name.lower()}
        queue: list[tuple[Chunk, int]] = [(root, 0)]

        while queue:
            cls, depth = queue.pop(0)
            for method in self._methods_of(cls.symbol_name, cls.file_path):
                key = method.symbol_name.lower()
                if key in seen_methods:
                    continue  # already defined nearer the subclass
                seen_methods.add(key)
                entries.append(OutlineEntry(
                    chunk=method,
                    defining_class=cls.symbol_name,
                    inherited=cls.id != root.id,
                ))
            if depth >= max_depth:
                continue
            for base in cls.base_classes:
                if base.lower() in visited:
                    continue
                visited.add(base.lower())
                resolved = self._find_class(base)
                if resolved is not None:
                    queue.append((resolved, depth + 1))
                else:
                    external.append(base)

        return ClassOutline(
            class_chunk=root,
            entries=entries,
            external_bases=external,
            subclasses=self._subclasses_of(root.symbol_name),
        )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        file_path=row["file_path"],
        symbol_name=row["symbol_name"],
        symbol_type=row["symbol_type"],
        parent_class=row["parent_class"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        docstring=row["docstring"],
        text=row["text"],
        embed_text=row["embed_text"],
        base_classes=_load_bases(row),
    )


def _load_bases(row: sqlite3.Row) -> list[str]:
    """Old rows predate the column, and ALTER TABLE leaves them at '[]'."""
    try:
        raw = row["base_classes"]
    except (IndexError, KeyError):
        return []
    return json.loads(raw) if raw else []
