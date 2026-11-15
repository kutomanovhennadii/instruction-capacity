"""Mirror tests of the alignment of words with the fewest edits."""

from iec.answer_validator.alignment.edit_alignment import edit_alignment


def test_edit_alignment_marks_the_lost_word() -> None:
    assert edit_alignment(["a", "c"], ["a", "b", "c"]) == [(0, 0), (None, 1), (1, 2)]
