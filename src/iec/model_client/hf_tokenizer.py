"""Hugging Face adapter behind the ``Tokenizer`` contract for open-weights models."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from iec.infrastructure.exceptions import ProviderError

HF_PROVIDER = "hf"


@dataclass(frozen=True, slots=True)
class HFTokenizer:
    """A Hugging Face tokenizer (Llama, Qwen and the like) behind the core's contract.

    Implements ``Tokenizer``. The given string is checked as is, without canonicalization:
    the probe form (in particular, the leading space) is chosen by the caller. Special tokens
    are disabled in the check (``add_special_tokens=False``) so that BOS/EOS do not inflate
    the count. The ``PreTrainedTokenizerBase`` object is a provider detail and does not leave
    the ``model_client`` layer.
    """

    name: str
    _tokenizer: PreTrainedTokenizerBase

    def is_single_token(self, word: str) -> bool:
        """Return whether the HF tokenizer encodes the given string as exactly one token."""
        return len(self._tokenizer.encode(word, add_special_tokens=False)) == 1

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text without special tokens."""
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def build_hf_tokenizer(model_name: str) -> HFTokenizer:
    """Load the HF tokenizer of a model and wrap it in the contract.

    Args:
        model_name: Path or name of the model on the Hugging Face Hub
            (for example ``"meta-llama/Llama-3.1-8B"``, ``"gpt2"``).

    Returns:
        A tokenizer with the versioned name ``"hf/<model_name>"``.

    Raises:
        ProviderError: if the tokenizer is not found or cannot be loaded - for example,
            there is no access to the Hub or to a private model
            (``error_code="HF_LOAD_FAILED"``); the original cause is kept.

    Side effects:
        The first load of a model downloads the tokenizer files from the Hugging Face Hub
        into the local cache and needs the network.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as cause:
        raise ProviderError.from_cause(
            f"failed to load HuggingFace tokenizer for '{model_name}'",
            cause,
            error_code="HF_LOAD_FAILED",
        ) from cause
    return HFTokenizer(name=f"{HF_PROVIDER}/{model_name}", _tokenizer=tokenizer)
