from unittest.mock import MagicMock, patch, call
import tempfile
from storage.chunk import make_chunk
from storage.db import DB
from indexer.embedder import embed_chunks, BATCH_SIZE


def _make_chunks(n: int) -> list:
    return [
        make_chunk(f"file_{i}.py", f"fn_{i}", "function", None, i, i + 5, None, f"def fn_{i}(): pass")
        for i in range(n)
    ]


def _fake_embedding(dim: int = 1536) -> list[float]:
    return [0.1] * dim


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_inserts_embeddings(mock_embed, tmp_path):
    mock_embed.return_value = [[0.1] * 1536]
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(1)
    rowid = db.insert_chunk(chunks[0])
    embed_chunks(chunks, db)
    assert db.has_embedding(rowid)


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_skips_existing(mock_embed, tmp_path):
    mock_embed.return_value = [[0.1] * 1536]
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(1)
    rowid = db.insert_chunk(chunks[0])
    db.insert_embedding(rowid, [0.1] * 1536)  # pre-insert

    embed_chunks(chunks, db)
    mock_embed.assert_not_called()


@patch("indexer.embedder._openai_embed")
def test_embed_chunks_batches(mock_embed, tmp_path):
    n = BATCH_SIZE + 1
    mock_embed.return_value = [[0.1] * 1536] * BATCH_SIZE
    db = DB(str(tmp_path / "test.db"))
    chunks = _make_chunks(n)
    for c in chunks:
        db.insert_chunk(c)

    def side_effect(texts):
        return [[0.1] * 1536] * len(texts)
    mock_embed.side_effect = side_effect

    embed_chunks(chunks, db)
    assert mock_embed.call_count == 2  # ceil(n / BATCH_SIZE)
