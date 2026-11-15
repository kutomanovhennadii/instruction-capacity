"""The strategies of deciding a gap between anchors and the rule that chooses one by size.

Two strategies decide a gap and give the same decision: the small one goes line by line, the large
one by whole matrices. The choice is a matter of speed only: on a few lines the matrices only cost
time, and the large strategy holds only reference lines that fit the bits of a machine word.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from iec.answer_validator.alignment.gaps.gap_decision import GapDecision
from iec.answer_validator.alignment.gaps.large_gap import LONGEST_MASKED_LINE, LargeGapResolver
from iec.answer_validator.alignment.gaps.small_gap import SmallGapResolver

#: A gap of at most this many pairs of lines goes the plain way: on a few lines the matrices
#: only cost time. The value changes the speed, not the result.
SMALL_GAP_PAIRS = 16


class GapResolver(Protocol):
    """A strategy that decides a gap between anchors: what to pair, what is deleted, inserted.

    An answer line that exactly copies a reference line is an insertion: it repeats its line off
    its place. The other lines are paired with the gap lines so that as many words as possible
    match; every pairing must pass the copy threshold. The value of a pairing is counted in half
    words: ``2·matched``, three less for a glue and a split. An answer line that passes the
    threshold in no pairing is an insertion; a reference line without a pair is a deletion.
    Matched words: by positions in a pair of one answer line and one reference line of equal
    length, otherwise the length of the longest common subsequence.
    """

    def resolve(
        self,
        gap_reference: Sequence[int],
        gap_answer: Sequence[int],
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        copies: frozenset[tuple[str, ...]],
    ) -> GapDecision:
        """Decide a gap between anchors.

        Args:
            gap_reference: The reference lines of the gap, in a row.
            gap_answer: The answer lines of the gap in order, without wrapper lines.
            reference_signatures: The canonical form of the reference lines - the source lines.
            answer_signatures: The canonical form of the answer lines.
            copies: The canonical form of all reference lines.

        Returns:
            The decisions of the gap and the pairings with the share of matched words of the
            shorter side.
        """
        ...


class SizedGapResolver:
    """Chooses the strategy of a gap by its size; both strategies give the same decision.

    A gap of at most :data:`SMALL_GAP_PAIRS` pairs of lines, or a gap with a reference line longer
    than the large strategy holds, goes to the small strategy; any other gap to the large one.
    """

    def __init__(self, small: GapResolver, large: GapResolver) -> None:
        """Remember the two strategies.

        Args:
            small: The strategy of a few lines and of long lines, line by line.
            large: The strategy of a large gap of short lines, by whole matrices.
        """
        self._small = small
        self._large = large

    def resolve(
        self,
        gap_reference: Sequence[int],
        gap_answer: Sequence[int],
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        copies: frozenset[tuple[str, ...]],
    ) -> GapDecision:
        """Decide a gap by the strategy its size calls for; the arguments are of ``GapResolver``."""
        strategy = (
            self._small
            if len(gap_reference) * len(gap_answer) <= SMALL_GAP_PAIRS
            or max(map(len, map(reference_signatures.__getitem__, gap_reference)))
            > LONGEST_MASKED_LINE
            else self._large
        )
        return strategy.resolve(
            gap_reference, gap_answer, reference_signatures, answer_signatures, copies
        )


#: The strategy of the alignment: small gaps line by line, large gaps of short lines by matrices.
DEFAULT_GAP_RESOLVER: GapResolver = SizedGapResolver(SmallGapResolver(), LargeGapResolver())
