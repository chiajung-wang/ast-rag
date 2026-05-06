from __future__ import annotations
import sqlite3
import struct
from storage.chunk import Chunk, make_chunk

EMBEDDING_DIM = 1536


def _serialize(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


class DB:
    def __init__(self, path: str = "index.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(f"""
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
                embed_text  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_symbol
                ON chunks(lower(symbol_name));
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding FLOAT[{EMBEDDING_DIM}]
            );
        """)
        self.conn.commit()

    def insert_chunk(self, chunk: Chunk) -> int:
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO chunks
                (id, file_path, symbol_name, symbol_type, parent_class,
                 line_start, line_end, docstring, text, embed_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id, chunk.file_path, chunk.symbol_name, chunk.symbol_type,
                chunk.parent_class, chunk.line_start, chunk.line_end,
                chunk.docstring, chunk.text, chunk.embed_text,
            ),
        )
        self.conn.commit()
        if cur.rowcount == 0:
            row = self.conn.execute(
                "SELECT rowid FROM chunks WHERE id = ?", (chunk.id,)
            ).fetchone()
            return row["rowid"]
        return cur.lastrowid

    def has_embedding(self, rowid: int) -> bool:
        return self.conn.execute(
            "SELECT rowid FROM vec_chunks WHERE rowid = ?", (rowid,)
        ).fetchone() is not None

    def insert_embedding(self, rowid: int, embedding: list[float]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
            (rowid, _serialize(embedding)),
        )
        self.conn.commit()

    def vector_search(self, embedding: list[float], k: int = 10) -> list[Chunk]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.file_path, c.symbol_name, c.symbol_type, c.parent_class,
                   c.line_start, c.line_end, c.docstring, c.text, c.embed_text
            FROM vec_chunks v
            JOIN chunks c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (_serialize(embedding), k),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def symbol_lookup(self, name: str) -> Chunk | None:
        row = self.conn.execute(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text
            FROM chunks WHERE lower(symbol_name) = lower(?) LIMIT 1
            """,
            (name,),
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def chunk_exists_at(self, file_path: str, line_start: int, line_end: int) -> bool:
        return self.conn.execute(
            """
            SELECT 1 FROM chunks
            WHERE file_path = ? AND line_start <= ? AND line_end >= ?
            LIMIT 1
            """,
            (file_path, line_start, line_end),
        ).fetchone() is not None

    def all_chunks(self) -> list[Chunk]:
        rows = self.conn.execute(
            """
            SELECT id, file_path, symbol_name, symbol_type, parent_class,
                   line_start, line_end, docstring, text, embed_text
            FROM chunks
            """
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def all_symbol_names(self) -> set[str]:
        rows = self.conn.execute("SELECT symbol_name FROM chunks").fetchall()
        return {r["symbol_name"] for r in rows}


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
    )
