"""The empty answer: an answered point whose answer holds no word."""

from __future__ import annotations

from iec.contracts import PointConditions, PointRecord, PointStatus, has_no_words, is_empty_answer

_CONDITIONS = PointConditions(D=4, n=1, m=1, seed_rules=1, seed_text=1, seed_order=0)


def _record(answer: str | None, status: PointStatus = PointStatus.OK) -> PointRecord:
    return PointRecord(
        conditions=_CONDITIONS,
        rules=(("lake", "cloud"),),
        source_text="lake stone tree hill",
        answer=answer,
        status=status,
        prompt_tokens=None,
        response_tokens=None,
    )


def test_a_text_without_a_word_is_empty() -> None:
    assert has_no_words(None) and has_no_words("") and has_no_words(" \n\t")
    assert not has_no_words("cloud")


def test_only_an_answered_point_without_a_word_is_an_empty_answer() -> None:
    assert is_empty_answer(_record("")) and is_empty_answer(_record(" \n"))
    assert not is_empty_answer(_record("cloud stone tree hill"))
    assert not is_empty_answer(_record(None, PointStatus.API_ERROR))
    # An answered point without an answer is a broken record, not an empty answer.
    assert not is_empty_answer(_record(None))


def test_no_words_is_judged_as_by_splitting_into_words() -> None:
    # Every white space character of Python lies below U+3100: the last of them is U+3000.
    for code in range(0x3100):
        character = chr(code)
        for text in (character, character * 3, f" {character} ", f"a{character}"):
            assert has_no_words(text) == (not text.split()), repr(text)
    assert has_no_words("") and has_no_words(None)
