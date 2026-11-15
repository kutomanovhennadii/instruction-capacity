"""Mirror tests of the rules of a gap decision."""

from iec.answer_validator.alignment.aligned_answer import PAIRING_GLUE, PAIRING_ONE, PAIRING_SPLIT
from iec.answer_validator.alignment.gaps.gap_decision import is_copy, pairing_value


def test_a_pair_is_a_copy_only_above_half_of_the_shorter_side() -> None:
    assert is_copy(3, 5, 9) is True
    assert is_copy(2, 4, 9) is False


def test_a_glue_and_a_split_cost_three_half_words() -> None:
    assert pairing_value(PAIRING_ONE, 4) == 8
    assert pairing_value(PAIRING_GLUE, 4) == 5
    assert pairing_value(PAIRING_SPLIT, 4) == 5
