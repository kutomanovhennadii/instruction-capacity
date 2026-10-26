"""Mirror tests of the vocabulary build core on mock tokenizers.

The word source is substituted explicitly (``word_source``), so the tests do not touch NLTK
and check exactly the pipeline of single-token status and intersection (no network).
"""

import logging
from collections.abc import Iterator

import pytest

from iec.contracts import Vocab
from iec.infrastructure.exceptions import VocabError
from iec.infrastructure.logging import PACKAGE_LOGGER_NAME
from iec.model_client import MockTokenizer
from iec.vocab import VocabBuildConfig, VocabBuilder


@pytest.fixture(autouse=True)
def isolated_package_logger() -> Iterator[None]:
    """Reset the package log handlers so that a new one binds to the capsys stream."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    logger.handlers.clear()
    try:
        yield
    finally:
        logger.handlers[:] = previous_handlers

# Words of different length: short - 4 letters, mid - 8 letters, long - 12 letters.
WORDS = {"lake", "mountains", "flabbergast"}  # 4, 9, 11 characters


def _config(vocab_id: str = "v_test") -> VocabBuildConfig:
    """Build a minimal config (real tokenizers are not used)."""
    return VocabBuildConfig.model_validate(
        {
            "vocab_id": vocab_id,
            "tokenizer_specs": [{"provider": "mock", "name": "mock"}],
        }
    )


def _source(words: set[str]):
    """Return a source function giving a fixed set of words."""
    return lambda _config: set(words)


def test_build_returns_vocab_with_matching_hash() -> None:
    builder = VocabBuilder(
        _config(),
        [MockTokenizer(name="mock/wide", single_token_max_chars=20)],
        word_source=_source(WORDS),
    )

    vocab = builder.build()

    assert isinstance(vocab, Vocab)
    assert vocab.words == frozenset(WORDS)
    assert vocab.hash == Vocab.compute_hash(WORDS)
    assert vocab.vocab_id == "v_test"
    assert vocab.tokenizer_names == ("mock/wide",)


def test_intersection_is_narrower_than_either_set() -> None:
    # The narrow tokenizer passes only words of <= 4 characters, the wide one <= 10.
    narrow = MockTokenizer(name="mock/narrow", single_token_max_chars=4)
    wide = MockTokenizer(name="mock/wide", single_token_max_chars=10)
    builder = VocabBuilder(_config(), [narrow, wide], word_source=_source(WORDS))

    vocab = builder.build()

    # "lake" (4) passes both; "mountains" (9) only the wide one; "flabbergast" (11) none.
    assert vocab.words == frozenset({"lake"})
    assert vocab.size == 1


def test_empty_source_raises_vocab_error() -> None:
    builder = VocabBuilder(
        _config(),
        [MockTokenizer()],
        word_source=_source(set()),
    )

    with pytest.raises(VocabError) as excinfo:
        builder.build()

    assert excinfo.value.error_code == "VOCAB_EMPTY_SOURCE"


def test_empty_intersection_raises_vocab_error() -> None:
    # The tokenizer rejects every word (threshold 0) -> an empty intersection.
    builder = VocabBuilder(
        _config(),
        [MockTokenizer(name="mock/zero", single_token_max_chars=0)],
        word_source=_source(WORDS),
    )

    with pytest.raises(VocabError) as excinfo:
        builder.build()

    assert excinfo.value.error_code == "VOCAB_EMPTY_INTERSECTION"


def test_no_tokenizers_raises_vocab_error() -> None:
    with pytest.raises(VocabError) as excinfo:
        VocabBuilder(_config(), [], word_source=_source(WORDS))

    assert excinfo.value.error_code == "VOCAB_NO_TOKENIZERS"


def test_hash_is_reproducible_across_builds() -> None:
    def fresh_builder() -> VocabBuilder:
        return VocabBuilder(
            _config(),
            [MockTokenizer(single_token_max_chars=20)],
            word_source=_source(WORDS),
        )

    first = fresh_builder().build()
    second = fresh_builder().build()

    assert first.hash == second.hash


def test_single_token_probe_uses_leading_space() -> None:
    # The probe goes with a leading space: in a BPE vocabulary the unit is the word after a
    # separator. The word in the ready vocabulary stays without the space.
    seen: list[str] = []

    class SpyTokenizer:
        name = "spy"

        def is_single_token(self, word: str) -> bool:
            seen.append(word)
            return True

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    builder = VocabBuilder(_config(), [SpyTokenizer()], word_source=_source({"lake"}))

    vocab = builder.build()

    assert seen == [" lake"]
    assert vocab.words == frozenset({"lake"})


def test_word_single_token_only_with_space_is_kept() -> None:
    # Regression on real BPE behaviour: "sword" is two tokens bare and one token with a
    # space. The vocabulary must keep such a word.
    class SpaceOnlyTokenizer:
        name = "space-only"

        def is_single_token(self, word: str) -> bool:
            return word.startswith(" ")

        def count_tokens(self, text: str) -> int:
            return len(text.split())

    builder = VocabBuilder(_config(), [SpaceOnlyTokenizer()], word_source=_source({"sword"}))

    assert builder.build().words == frozenset({"sword"})


def test_per_tokenizer_sizes_are_logged(capsys: pytest.CaptureFixture[str]) -> None:
    # The package logger does not propagate records (propagate=False), so stderr is read,
    # as in the mock provider tests, not caplog.
    builder = VocabBuilder(
        _config(),
        [MockTokenizer(name="mock/wide", single_token_max_chars=20)],
        word_source=_source(WORDS),
    )

    builder.build()

    err = capsys.readouterr().err
    assert "mock/wide" in err and "single-token words" in err
    assert "intersection size" in err
