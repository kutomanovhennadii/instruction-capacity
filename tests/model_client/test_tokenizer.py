"""Mirror tests of the contract of a model tokenizer."""

from iec.model_client import Tokenizer


class _WordTokenizer:
    """A tokenizer that satisfies the contract: a name, the single-token check and the count."""

    name = "words"

    def is_single_token(self, word: str) -> bool:
        """Return whether the string is one word."""
        return len(word.split()) == 1

    def count_tokens(self, text: str) -> int:
        """Count the words of the text."""
        return len(text.split())


class _WithoutCount:
    """A tokenizer without the count of tokens."""

    name = "partial"

    def is_single_token(self, word: str) -> bool:
        """Return whether the string is one word."""
        return len(word.split()) == 1


def test_an_object_with_the_three_members_is_a_tokenizer() -> None:
    assert isinstance(_WordTokenizer(), Tokenizer)


def test_an_object_without_the_count_is_not_a_tokenizer() -> None:
    assert not isinstance(_WithoutCount(), Tokenizer)
