"""Mirror tests of the small gap decided line by line."""

from iec.answer_validator.alignment.aligned_answer import GROUP_INSERTED, PAIRING_ONE
from iec.answer_validator.alignment.gaps.gap_decision import Group
from iec.answer_validator.alignment.gaps.small_gap import SmallGapResolver, lcs_length


def test_lcs_length_counts_the_longest_common_subsequence() -> None:
    assert lcs_length(["a", "b", "c", "d"], ["a", "c", "d", "b"]) == 3


def test_a_rewritten_line_is_paired_and_a_copy_is_inserted() -> None:
    reference = [("a", "b", "c", "d")]
    answer = [("a", "b", "x", "d"), ("a", "b", "c", "d")]

    groups, pairings = SmallGapResolver().resolve(
        [0], [0, 1], reference, answer, frozenset(reference)
    )

    assert groups == [Group(PAIRING_ONE, (0,), (0,)), Group(GROUP_INSERTED, (), (1,))]
    assert pairings == [(PAIRING_ONE, 0.75)]
