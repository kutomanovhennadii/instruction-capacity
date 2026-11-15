"""Mirror tests of the accumulator of places."""

from iec.answer_validator.alignment.aligned_answer import (
    GROUP_ANCHOR,
    GROUP_CODES,
    GROUP_INSERTED,
    KIND_INSERTED,
    KIND_MATCHED,
)
from iec.answer_validator.alignment.gaps.gap_decision import Group
from iec.answer_validator.alignment.place_builder import PlaceBuilder


def _builder() -> PlaceBuilder:
    """A builder of two lines of two words, the answer equal to the source."""
    lines = (("lake", "stone"), ("tree", "fish"))
    return PlaceBuilder(lines, lines, list(lines), [0, 2], {0: 0, 1: 2})


def test_consecutive_anchors_join_into_one_run() -> None:
    builder = _builder()
    builder.add_anchor(0, 0)
    builder.add_anchor(1, 1)

    kinds, references, answers, origins = builder.arrays()

    assert kinds.tolist() == [KIND_MATCHED] * 4
    assert references.tolist() == [0, 1, 2, 3]
    assert answers.tolist() == [0, 1, 2, 3]
    assert origins.tolist() == [GROUP_CODES[GROUP_ANCHOR]] * 4
    assert builder.unmatched == 0


def test_an_inserted_line_counts_its_words_as_unmatched() -> None:
    builder = _builder()
    builder.add_group(Group(GROUP_INSERTED, (), (1,)))

    kinds, references, answers, _ = builder.arrays()

    assert kinds.tolist() == [KIND_INSERTED, KIND_INSERTED]
    assert references.tolist() == [-1, -1]
    assert answers.tolist() == [2, 3]
    assert builder.unmatched == 2
