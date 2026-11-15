"""Mirror tests of the task source text contract."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import SourceText


def _make_text(words: tuple[str, ...], *, words_per_line: int = 3) -> SourceText:
    """Assemble a word stream with a dummy markup of two checked replacements."""
    return SourceText(
        words=words,
        words_per_line=words_per_line,
        replacement_positions=(1, 3),
        len_tokens=len(words),
        tokenizer_name="mock",
        seed_text=42,
    )


def test_fields_are_stored_as_given() -> None:
    text = _make_text(("tree", "dog", "fish", "lake", "sun"))

    assert text.words == ("tree", "dog", "fish", "lake", "sun")
    assert text.words_per_line == 3
    assert text.replacement_positions == (1, 3)
    assert text.len_tokens == 5
    assert text.tokenizer_name == "mock"
    assert text.seed_text == 42


def test_text_assembles_lines_with_short_last_line() -> None:
    text = _make_text(("tree", "dog", "fish", "lake", "sun"))

    assert text.text == "tree dog fish\nlake sun"


def test_text_with_exact_multiple_has_no_trailing_line() -> None:
    text = _make_text(("tree", "dog", "fish", "lake", "sun", "cat"))

    assert text.text == "tree dog fish\nlake sun cat"


def test_text_single_line_when_words_fit() -> None:
    text = _make_text(("tree", "dog"), words_per_line=10)

    assert text.text == "tree dog"


def test_replacement_count_equals_marked_positions() -> None:
    text = _make_text(("tree", "dog", "fish", "lake", "sun"))

    assert text.replacement_count == 2
    assert text.replacement_count == len(text.replacement_positions)


def test_as_dict_is_json_serializable_round_trip() -> None:
    text = _make_text(("tree", "dog", "fish", "lake", "sun"))

    snapshot = text.as_dict()

    assert snapshot == {
        "words": ["tree", "dog", "fish", "lake", "sun"],
        "words_per_line": 3,
        "replacement_positions": [1, 3],
        "len_tokens": 5,
        "tokenizer_name": "mock",
        "seed_text": 42,
        "replacement_count": 2,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_contract_is_frozen() -> None:
    text = _make_text(("tree", "dog"))

    with pytest.raises(FrozenInstanceError):
        text.len_tokens = 99  # type: ignore[misc]
