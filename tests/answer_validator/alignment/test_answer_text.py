"""Mirror tests of the text forms of an answer."""

from iec.answer_validator.alignment.answer_text import normalize_output, split_lines


def test_only_the_trailing_whitespace_is_stripped() -> None:
    assert normalize_output("  Lake stone \n\n") == "  Lake stone"


def test_an_empty_line_stays_a_line_without_words() -> None:
    assert split_lines("lake  stone\n\ntree") == (("lake", "stone"), (), ("tree",))
