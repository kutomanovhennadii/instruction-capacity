"""Mirror tests of the choice of the strategy of a gap by its size."""

from collections.abc import Sequence

from iec.answer_validator.alignment.gaps.gap_decision import GapDecision
from iec.answer_validator.alignment.gaps.gap_resolver import SMALL_GAP_PAIRS, SizedGapResolver
from iec.answer_validator.alignment.gaps.large_gap import LONGEST_MASKED_LINE


class _Recorder:
    """A strategy that counts its calls and decides nothing."""

    def __init__(self) -> None:
        self.calls = 0

    def resolve(
        self,
        gap_reference: Sequence[int],
        gap_answer: Sequence[int],
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        copies: frozenset[tuple[str, ...]],
    ) -> GapDecision:
        self.calls += 1
        return [], []


def _choose(reference_lines: int, answer_lines: int, line_length: int) -> tuple[int, int]:
    """Decide a gap of the given size and return the calls of the small and the large strategy."""
    small, large = _Recorder(), _Recorder()
    signatures = [("w",) * line_length] * max(reference_lines, answer_lines)
    SizedGapResolver(small, large).resolve(
        list(range(reference_lines)), list(range(answer_lines)), signatures, signatures, frozenset()
    )
    return small.calls, large.calls


def test_a_gap_of_few_pairs_goes_the_plain_way() -> None:
    assert _choose(1, SMALL_GAP_PAIRS, 3) == (1, 0)


def test_a_gap_of_many_pairs_of_short_lines_goes_by_matrices() -> None:
    assert _choose(1, SMALL_GAP_PAIRS + 1, 3) == (0, 1)


def test_a_gap_with_a_line_too_long_for_a_machine_word_goes_the_plain_way() -> None:
    assert _choose(1, SMALL_GAP_PAIRS + 1, LONGEST_MASKED_LINE + 1) == (1, 0)
