from __future__ import annotations
import re
from rank_bm25 import BM25Okapi
from storage.chunk import Chunk


def _tokenize(text: str) -> list[str]:
    tokens = []
    for word in re.findall(r'[A-Za-z0-9_]+', text):
        snake_parts = [p for p in word.split('_') if p]
        if len(snake_parts) > 1:
            tokens.extend(p.lower() for p in snake_parts)
            tokens.append(word.lower())
        else:
            s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', word)
            s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
            parts = s.split()
            if len(parts) > 1:
                tokens.extend(p.lower() for p in parts)
                tokens.append(word.lower())
            else:
                tokens.append(word.lower())
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        tokenized = [_tokenize(c.embed_text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if chunks else None

    def search(self, query: str, k: int = 10) -> list[Chunk]:
        if not self._chunks or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top]

    @classmethod
    def from_db(cls, db) -> "BM25Index":
        return cls(db.all_chunks())
