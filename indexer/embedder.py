from __future__ import annotations
import math
from dotenv import load_dotenv
from openai import OpenAI
from storage.chunk import Chunk
from storage.db import DB, EMBED_MODEL_KEY
import provider

load_dotenv()

BATCH_SIZE = 100
MAX_CHARS = 24_000  # ~6k tokens @ 4 chars/token — safely under 8192 limit
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=provider.BASE_URL, api_key=provider.api_key())
    return _client


def _openai_embed(texts: list[str]) -> list[list[float]]:
    response = _get_client().embeddings.create(
        model=provider.embed_model(),
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_chunks(chunks: list[Chunk], db: DB) -> None:
    # Record which model produced these vectors so a later query against a
    # different EMBED_MODEL fails loudly instead of returning wrong neighbours.
    db.set_meta(EMBED_MODEL_KEY, provider.embed_model())

    pairs: list[tuple[Chunk, int]] = []
    for chunk in chunks:
        rowid = db.insert_chunk(chunk)
        if not db.has_embedding(rowid):
            pairs.append((chunk, rowid))

    if not pairs:
        return

    num_batches = math.ceil(len(pairs) / BATCH_SIZE)
    for i in range(num_batches):
        batch = pairs[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        texts = [c.embed_text[:MAX_CHARS] for c, _ in batch]
        embeddings = _openai_embed(texts)
        for (_, rowid), embedding in zip(batch, embeddings):
            db.insert_embedding(rowid, embedding)
        print(f"  embedded batch {i + 1}/{num_batches} ({len(batch)} chunks)")
