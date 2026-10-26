"""The tiktoken adapter behind the ``Tokenizer`` contract for OpenAI models."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken

from iec.infrastructure.exceptions import ProviderError

TIKTOKEN_PROVIDER = "tiktoken"


@dataclass(frozen=True, slots=True)
class TiktokenTokenizer:
    """The OpenAI tokenizer (GPT-3.5, GPT-4o, GPT-4o-mini and so on) behind the project contract.

    Implements ``Tokenizer``. The given string is checked as is, without canonicalization:
    the probe form (in particular, the leading space) is chosen by the caller. The
    ``tiktoken.Encoding`` object is a provider detail and does not leave the model layer.
    """

    name: str
    _encoding: tiktoken.Encoding

    def is_single_token(self, word: str) -> bool:
        """Return whether tiktoken encodes the given string as exactly one token."""
        return len(self._encoding.encode(word)) == 1

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text by the rules of tiktoken."""
        return len(self._encoding.encode(text))


def build_tiktoken_tokenizer(model_name: str) -> TiktokenTokenizer:
    """Load the tiktoken encoder of an OpenAI model and wrap it in the contract.

    Args:
        model_name: OpenAI model name (for example ``"gpt-4o"``, ``"gpt-3.5-turbo"``).

    Returns:
        A tokenizer with the versioned name ``"tiktoken/<model_name>"``.

    Raises:
        ProviderError: if tiktoken cannot find the encoding for the model
            (``error_code="TIKTOKEN_LOAD_FAILED"``); the original cause is kept.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as cause:
        raise ProviderError.from_cause(
            f"failed to load tiktoken encoding for '{model_name}'",
            cause,
            error_code="TIKTOKEN_LOAD_FAILED",
        ) from cause
    return TiktokenTokenizer(name=f"{TIKTOKEN_PROVIDER}/{model_name}", _encoding=encoding)
