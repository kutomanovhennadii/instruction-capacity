"""Mirror tests of the counting tokenizer of a vocabulary build."""

import logging

import pytest

from iec.model_client import MockTokenizer, Tokenizer
from iec.vocab import CountingTokenizer
from iec.vocab.counting_tokenizer import PROGRESS_EVERY


def test_probes_and_single_token_answers_are_counted() -> None:
    counter = CountingTokenizer(MockTokenizer())

    answers = [counter.is_single_token(probe) for probe in (" lake", " lake stone", " cloud")]

    assert answers == [True, False, True]
    assert (counter.probes, counter.single) == (3, 2)


def test_the_counter_answers_as_the_wrapped_tokenizer() -> None:
    inner = MockTokenizer(name="mock/inner")
    counter = CountingTokenizer(inner)

    assert isinstance(counter, Tokenizer)
    assert counter.name == "mock/inner"
    assert counter.count_tokens("lake stone cloud") == inner.count_tokens("lake stone cloud")


def test_progress_is_logged_every_progress_step(
    isolated_package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    counter = CountingTokenizer(MockTokenizer(name="mock/slow"))

    for _ in range(PROGRESS_EVERY):
        counter.is_single_token(" lake")

    err = capsys.readouterr().err
    assert f"tokenizer mock/slow: {PROGRESS_EVERY} probes, {PROGRESS_EVERY} single" in err
