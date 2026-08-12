"""OpenRouter is the only provider. Chat and embeddings both go through it.

One key, one base URL, one place to change them.
"""
from __future__ import annotations
import os

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_VAR = "OPENROUTER_API_KEY"

DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
DEFAULT_AGENT_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.6"


def api_key() -> str:
    key = os.environ.get(API_KEY_VAR, "")
    if not key:
        raise RuntimeError(
            f"{API_KEY_VAR} is not set. Copy .env.example to .env and add your "
            "OpenRouter key (https://openrouter.ai/keys)."
        )
    return key


def embed_model() -> str:
    """Embedding model for indexing and queries.

    Changing this invalidates the index: two models do not share a vector
    space. `storage.db` records the model used at index time and refuses a
    query on mismatch, so the failure is loud rather than a quiet drop in
    retrieval quality.
    """
    return os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL)


def agent_model() -> str:
    return os.environ.get("AGENT_MODEL", DEFAULT_AGENT_MODEL)


def judge_model() -> str:
    return os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
