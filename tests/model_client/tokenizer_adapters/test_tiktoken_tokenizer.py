"""Mirror tests of the tiktoken adapter.

The OpenAI encodings (``o200k_base`` for gpt-4o) ship with the tiktoken package, so the
tests are deterministic and need no network.
"""

import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.model_client import Tokenizer, build_tiktoken_tokenizer

MODEL_NAME = "gpt-4o"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Build the adapter once per module (the encoding is cached in the package)."""
    return build_tiktoken_tokenizer(MODEL_NAME)


def test_satisfies_tokenizer_protocol(tokenizer: Tokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)


def test_name_is_versioned(tokenizer: Tokenizer) -> None:
    assert tokenizer.name == f"tiktoken/{MODEL_NAME}"


def test_short_word_is_single_token(tokenizer: Tokenizer) -> None:
    assert tokenizer.is_single_token("lake") is True


def test_long_word_is_not_single_token(tokenizer: Tokenizer) -> None:
    assert tokenizer.is_single_token("incomprehensibilities") is False


def test_count_tokens_returns_positive_int(tokenizer: Tokenizer) -> None:
    count = tokenizer.count_tokens("hello world")

    assert isinstance(count, int)
    assert count > 0


def test_unknown_model_raises_provider_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        build_tiktoken_tokenizer("definitely-not-a-real-model-xyz")

    assert excinfo.value.error_code == "TIKTOKEN_LOAD_FAILED"
    assert excinfo.value.__cause__ is not None
