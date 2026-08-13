"""Check that the configured embedding provider agrees with the built index.

Why this exists: embeddings from two models do not share a vector space, and
neither do the same model served by two providers if either quietly differs.
A mismatch does not raise anything. Retrieval just returns plausible, wrong
neighbours, and every eval number silently becomes meaningless.

`storage.db.assert_embed_model` catches the case where the *name* changed.
This catches the case where the name is the same but the vectors are not —
which is exactly the risk when switching provider, as this project did when
it moved from calling OpenAI directly to routing through OpenRouter.

What it does: takes chunks that already have stored vectors, re-embeds their
exact `embed_text` through the current provider, and compares. Identical
service means cosine ~1.0. The stored vectors are float32, so expect a
handful of decimal places of noise, never a difference in the third.

    python verify_embeddings.py            # 10 chunks
    python verify_embeddings.py --n 50     # more confidence, still cents

Exit code 0 means the index is safe to keep. 1 means re-run `make index`.
"""
from __future__ import annotations
import argparse
import math
import sys

from dotenv import load_dotenv

load_dotenv()

from indexer.corpus_config import DB_PATH  # noqa: E402
from indexer.embedder import MAX_CHARS, _openai_embed  # noqa: E402
from storage.db import DB, EMBED_MODEL_KEY  # noqa: E402
import provider  # noqa: E402

# float32 round-tripping alone costs ~1e-7. Anything above this threshold is
# the same service; anything below means a different model or a different
# serving path, and the index cannot be trusted.
COSINE_THRESHOLD = 0.9999


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="chunks to sample")
    args = parser.parse_args()

    db = DB(DB_PATH)
    model = provider.embed_model()
    recorded = db.get_meta(EMBED_MODEL_KEY)

    print(f"index      {DB_PATH}")
    print(f"built with {recorded or '(not recorded — index predates the guard)'}")
    print(f"current    {model} via {provider.BASE_URL}\n")

    sample = db.sample_embedded_chunks(args.n)
    if not sample:
        print("No embedded chunks found. Run `make index` first.")
        return 1

    # Same truncation the indexer applied, or the texts are not comparable.
    texts = [chunk.embed_text[:MAX_CHARS] for _, chunk in sample]
    try:
        fresh = _openai_embed(texts)
    except Exception as exc:  # noqa: BLE001 - report, don't traceback
        print(f"Could not reach the embedding provider:\n  {exc}")
        return 1

    stored_dim = len(db.get_embedding(sample[0][0]))
    if len(fresh[0]) != stored_dim:
        print(f"DIMENSION MISMATCH: index is {stored_dim}-d, {model} returns "
              f"{len(fresh[0])}-d.\nThe index cannot be reused. Run `make index`.")
        return 1

    scores = []
    for (rowid, chunk), new_vec in zip(sample, fresh):
        score = cosine(db.get_embedding(rowid), new_vec)
        scores.append(score)
        flag = "ok  " if score >= COSINE_THRESHOLD else "DIFF"
        print(f"  {flag} {score:.6f}  {chunk.symbol_name[:38]:38} {chunk.file_path}")

    worst = min(scores)
    mean = sum(scores) / len(scores)
    print(f"\n{len(scores)} chunks | mean {mean:.6f} | worst {worst:.6f} "
          f"| threshold {COSINE_THRESHOLD}")

    if worst >= COSINE_THRESHOLD:
        print("\nPASS — vectors match the index. Existing embeddings and every "
              "retrieval number measured against them remain valid.")
        return 0

    print("\nFAIL — the current provider returns different vectors from the ones "
          "in the index.\nQueries will be compared against a space they do not "
          "belong to, which degrades retrieval without raising anything.\n"
          "Run `make index` to rebuild, then re-run the ablation "
          "(`make eval-retrieval`) since its numbers no longer apply.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
