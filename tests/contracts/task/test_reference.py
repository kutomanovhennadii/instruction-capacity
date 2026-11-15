"""Mirror tests of the reference answer contract."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import Reference, SourceText


def _make_reference(words: tuple[str, ...], *, words_per_line: int = 3) -> Reference:
    """Assemble a reference word stream with the given line width."""
    return Reference(words=words, words_per_line=words_per_line)


def test_fields_are_stored_as_given() -> None:
    reference = _make_reference(("tree", "cat", "fish"))

    assert reference.words == ("tree", "cat", "fish")
    assert reference.words_per_line == 3


def test_text_assembles_lines_with_short_last_line() -> None:
    reference = _make_reference(("tree", "cat", "fish", "lake", "sun"))

    assert reference.text == "tree cat fish\nlake sun"


def test_text_with_exact_multiple_has_no_trailing_line() -> None:
    reference = _make_reference(("tree", "cat", "fish", "lake", "sun", "dog"))

    assert reference.text == "tree cat fish\nlake sun dog"


def test_text_single_line_when_words_fit() -> None:
    reference = _make_reference(("tree", "cat"), words_per_line=10)

    assert reference.text == "tree cat"


def test_text_matches_source_text_assembly_rule() -> None:
    """The reference and the source differ in words only, not in the split into lines."""
    words = ("tree", "dog", "fish", "lake", "sun")
    source = SourceText(
        words=words,
        words_per_line=3,
        replacement_positions=(1,),
        len_tokens=len(words),
        tokenizer_name="mock",
        seed_text=42,
    )
    reference = _make_reference(("tree", "cat", "fish", "lake", "sun"))

    assert reference.text.splitlines() == [
        "tree cat fish",
        "lake sun",
    ]
    assert [len(line.split()) for line in reference.text.splitlines()] == [
        len(line.split()) for line in source.text.splitlines()
    ]


def test_as_dict_is_json_serializable_round_trip() -> None:
    reference = _make_reference(("tree", "cat", "fish"))

    snapshot = reference.as_dict()

    assert snapshot == {
        "words": ["tree", "cat", "fish"],
        "words_per_line": 3,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_contract_is_frozen() -> None:
    reference = _make_reference(("tree", "cat"))

    with pytest.raises(FrozenInstanceError):
        reference.words_per_line = 99  # type: ignore[misc]
