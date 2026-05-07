from __future__ import annotations
from storage.chunk import Chunk


def rrf(
    bm25_results: list[Chunk],
    dense_results: list[Chunk],
    k: int = 60,
    top_n: int = 5,
) -> list[Chunk]:
    scores: dict[str, float] = {}
    id_to_chunk: dict[str, Chunk] = {}

    for rank, chunk in enumerate(bm25_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        id_to_chunk[chunk.id] = chunk

    for rank, chunk in enumerate(dense_results, 1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
        id_to_chunk[chunk.id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [id_to_chunk[chunk_id] for chunk_id, _ in ranked[:top_n]]
