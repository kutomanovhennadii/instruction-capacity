"""Mirror tests of the Hugging Face adapter.

The live check uses the ``Qwen/Qwen2.5-0.5B`` tokenizer - the same tokenizer as the
production Qwen, a member of the vocabulary set, and mechanically different from the
tiktoken BPE (not a duplicate of the tiktoken test). The tokenizer is downloaded from the
Hugging Face Hub on first access, so the live tests are marked ``network`` and are skipped
without the network rather than failing (no network). The error path of the factory needs no
network and is always checked.
"""

import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.model_client import Tokenizer
from iec.model_client.hf_tokenizer import build_hf_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Build the HF adapter once per module; without the network - skip the live tests."""
    try:
        return build_hf_tokenizer(MODEL_NAME)
    except ProviderError as error:
        pytest.skip(f"Hugging Face Hub is unavailable: {error}")


@pytest.mark.network
def test_satisfies_tokenizer_protocol(tokenizer: Tokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)


@pytest.mark.network
def test_name_is_versioned(tokenizer: Tokenizer) -> None:
    assert tokenizer.name == f"hf/{MODEL_NAME}"


@pytest.mark.network
def test_short_word_is_single_token(tokenizer: Tokenizer) -> None:
    assert tokenizer.is_single_token("lake") is True


@pytest.mark.network
def test_long_word_is_not_single_token(tokenizer: Tokenizer) -> None:
    assert tokenizer.is_single_token("incomprehensibilities") is False


@pytest.mark.network
def test_count_tokens_returns_positive_int(tokenizer: Tokenizer) -> None:
    count = tokenizer.count_tokens("hello world")

    assert isinstance(count, int)
    assert count > 0


@pytest.mark.network
def test_count_excludes_special_tokens(tokenizer: Tokenizer) -> None:
    # add_special_tokens=False in the adapter: the count must not be inflated by special tokens.
    assert tokenizer.count_tokens("lake") == 1


def test_unknown_model_raises_provider_error() -> None:
    with pytest.raises(ProviderError) as excinfo:
        build_hf_tokenizer("definitely-not-a-real-org/definitely-not-a-real-model-xyz")

    assert excinfo.value.error_code == "HF_LOAD_FAILED"
    assert excinfo.value.__cause__ is not None
