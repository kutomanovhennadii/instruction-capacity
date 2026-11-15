"""The small gap decided line by line: the plain table over the pairs of gap lines.

A gap of a few lines, or a gap with a reference line too long for the bits of a machine word,
goes this way. The pairs of lines are counted one by one with a cache, and the table is filled
cell by cell.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from iec.answer_validator.alignment.aligned_answer import PAIRING_GLUE, PAIRING_ONE, PAIRING_SPLIT
from iec.answer_validator.alignment.gaps.gap_decision import (
    GapDecision,
    decide_from_table,
    is_copy,
    pairing_value,
)


def lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    """Return the length of the longest common subsequence of two sequences of words."""
    previous = [0] * (len(second) + 1)
    for word in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if word == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


class SmallGapResolver:
    """The strategy of a small gap: the table over the pairs of gap lines, line by line.

    It decides any gap; the alignment gives it gaps of a few pairs of lines and gaps with a
    reference line longer than the large strategy holds.
    """

    def resolve(
        self,
        gap_reference: Sequence[int],
        gap_answer: Sequence[int],
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        copies: frozenset[tuple[str, ...]],
    ) -> GapDecision:
        """Decide a gap line by line; the arguments and the result are those of ``GapResolver``."""
        return _resolve_small_gap(
            gap_reference, gap_answer, reference_signatures, answer_signatures, copies
        )


def _resolve_small_gap(
    gap_reference: Sequence[int],
    gap_answer: Sequence[int],
    reference_signatures: Sequence[tuple[str, ...]],
    answer_signatures: Sequence[tuple[str, ...]],
    copies: frozenset[tuple[str, ...]],
) -> GapDecision:
    """Decide a gap line by line. The subsequence is computed only if the number of common words
    regardless of order passes the threshold: that number is never below the subsequence.
    """
    reference = list(gap_reference)
    cache: dict[tuple[str, tuple[tuple[str, ...], ...], tuple[int, ...]], int | None] = {}

    def matched(
        kind: str, answer_part: tuple[int, ...], reference_part: tuple[int, ...]
    ) -> int | None:
        """Return the matched words of a pairing, ``None`` if it does not pass the threshold."""
        key = (kind, tuple(answer_signatures[j] for j in answer_part), reference_part)
        if key not in cache:
            answer_words = [word for j in answer_part for word in answer_signatures[j]]
            reference_words = [word for i in reference_part for word in reference_signatures[i]]
            sizes = (len(answer_words), len(reference_words))
            if kind == PAIRING_ONE and sizes[0] == sizes[1]:
                count = sum(
                    1
                    for left, right in zip(answer_words, reference_words, strict=True)
                    if left == right
                )
            elif not is_copy(
                sum((Counter(answer_words) & Counter(reference_words)).values()), *sizes
            ):
                count = 0
            else:
                count = lcs_length(answer_words, reference_words)
            cache[key] = count if is_copy(count, *sizes) else None
        return cache[key]

    copied = [j for j in gap_answer if answer_signatures[j] in copies]
    candidates = [j for j in gap_answer if answer_signatures[j] not in copies]
    candidate_set = frozenset(candidates)

    def can_pair(j: int) -> bool:
        """Tell whether the answer line passes the copy threshold in some pairing of the gap."""
        if any(matched(PAIRING_ONE, (j,), (i,)) is not None for i in reference):
            return True
        if any(
            matched(PAIRING_GLUE, (j,), (reference[t], reference[t + 1])) is not None
            for t in range(len(reference) - 1)
        ):
            return True
        for first, second in ((j - 1, j), (j, j + 1)):
            if (
                first in candidate_set
                and second in candidate_set
                and any(
                    matched(PAIRING_SPLIT, (first, second), (i,)) is not None for i in reference
                )
            ):
                return True
        return False

    survivors = [j for j in candidates if reference and can_pair(j)]
    survivor_set = frozenset(survivors)
    lonely = [j for j in candidates if j not in survivor_set]

    def moves(a: int, b: int) -> list[tuple[str, int, int, int]]:
        """Return the pairings that end at the cell ``(a, b)`` of the table."""
        found: list[tuple[str, int, int, int]] = []
        if a >= 1 and b >= 1:
            count = matched(PAIRING_ONE, (survivors[b - 1],), (reference[a - 1],))
            if count is not None:
                found.append((PAIRING_ONE, 1, 1, count))
        if a >= 1 and b >= 2 and survivors[b - 1] == survivors[b - 2] + 1:
            count = matched(
                PAIRING_SPLIT, (survivors[b - 2], survivors[b - 1]), (reference[a - 1],)
            )
            if count is not None:
                found.append((PAIRING_SPLIT, 1, 2, count))
        if a >= 2 and b >= 1:
            count = matched(PAIRING_GLUE, (survivors[b - 1],), (reference[a - 2], reference[a - 1]))
            if count is not None:
                found.append((PAIRING_GLUE, 2, 1, count))
        return found

    rows, columns = len(reference), len(survivors)
    best = [[0] * (columns + 1) for _ in range(rows + 1)]
    for a in range(rows + 1):
        for b in range(columns + 1):
            if a == 0 and b == 0:
                continue
            here = max(best[a - 1][b] if a else -1, best[a][b - 1] if b else -1)
            for kind, step_a, step_b, count in moves(a, b):
                here = max(here, best[a - step_a][b - step_b] + pairing_value(kind, count))
            best[a][b] = here
    return decide_from_table(
        reference,
        survivors,
        (*copied, *lonely),
        lambda a, b: best[a][b],
        moves,
        reference_signatures,
        answer_signatures,
    )
