"""A tokenizer read from the ``tokenizer.json`` of a Hugging Face repository.

The adapter ``hf`` loads a tokenizer through ``transformers.AutoTokenizer``, and that one does not
load the tokenizer of Gemma 4 without the package protobuf. The file ``tokenizer.json`` is in the
repository, and the library ``tokenizers`` reads it directly: the same token vocabulary and the
same merge rules. Special tokens are not counted (``add_special_tokens=False``), as in the adapter
``hf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenizers import Tokenizer as TokenizerFile

from iec.model_client.tokenizer_adapters.library_loading import LibraryLoading

HUB_PROVIDER = "hfjson"
#: Loading of a ``tokenizer.json`` file. ``tokenizers`` downloads it through the Hugging Face Hub,
#: whose errors for a missing or private repository and an unavailable network subclass
#: ``OSError``.
HUB_LOADING = LibraryLoading(
    errors=(OSError,),
    error_code="HUB_TOKENIZER_LOAD_FAILED",
    message="failed to load tokenizer.json of '{name}'",
)


@dataclass(frozen=True, slots=True)
class HubTokenizer:
    """A tokenizer from ``tokenizer.json`` behind the ``Tokenizer`` contract; strings go as is."""

    name: str
    _tokenizer: Any

    def is_single_token(self, word: str) -> bool:
        """Return whether the string is encoded as exactly one token."""
        return len(self._tokenizer.encode(word, add_special_tokens=False).ids) == 1

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text without special tokens."""
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


def build_hub_tokenizer(repo_id: str) -> HubTokenizer:
    """Load the ``tokenizer.json`` of a repository and wrap it in the contract.

    Raises:
        ProviderError: ``HUB_TOKENIZER_LOAD_FAILED`` - the file is missing or the Hub is down.

    Side effects:
        The first load downloads the file into the Hugging Face cache and needs the network.
    """
    tokenizer = HUB_LOADING.load(lambda: TokenizerFile.from_pretrained(repo_id), repo_id)
    return HubTokenizer(name=f"{HUB_PROVIDER}/{repo_id}", _tokenizer=tokenizer)
