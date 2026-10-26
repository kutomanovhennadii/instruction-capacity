"""Universal contract of a model tokenizer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """A tokenizer that hides a particular library from the vocabulary builder.

    ``is_single_token`` checks the given string **as is**, adding or removing nothing: given
    ``" lake"`` it checks ``" lake"``. Which form of the word to probe is decided by the
    caller (``vocab_builder`` prepends a leading space, because in a BPE vocabulary and in
    the protocol prompt the word stands after a separator). Real BPE adapters must keep
    exactly these semantics and must not canonicalize the input on their own.
    """

    @property
    def name(self) -> str:
        """Return the versioned name of the tokenizer implementation."""
        ...

    def is_single_token(self, word: str) -> bool:
        """Return whether the given string is encoded as exactly one token (checked as is)."""
        ...

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text by the rules of the implementation."""
        ...
