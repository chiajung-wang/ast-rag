import os
from unittest.mock import patch

import pytest

import provider


def test_defaults_are_openrouter_slugs():
    """OpenRouter uses dots, not dashes: claude-haiku-4.5, not claude-haiku-4-5.
    A wrong slug 404s at request time and misses the eval price table."""
    assert provider.DEFAULT_AGENT_MODEL == "anthropic/claude-haiku-4.5"
    assert provider.DEFAULT_JUDGE_MODEL == "anthropic/claude-sonnet-4.6"
    assert provider.DEFAULT_EMBED_MODEL == "openai/text-embedding-3-small"


def test_base_url_is_openrouter():
    assert provider.BASE_URL == "https://openrouter.ai/api/v1"


def test_models_read_from_env():
    with patch.dict(os.environ, {"EMBED_MODEL": "x/y", "AGENT_MODEL": "a/b", "JUDGE_MODEL": "c/d"}):
        assert provider.embed_model() == "x/y"
        assert provider.agent_model() == "a/b"
        assert provider.judge_model() == "c/d"


def test_missing_key_raises_with_actionable_message():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            provider.api_key()
