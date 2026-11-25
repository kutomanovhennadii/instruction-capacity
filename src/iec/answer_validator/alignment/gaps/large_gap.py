"""The large gap decided by whole matrices: the same table and walk as on a small gap.

An answer that lost its structure has thousands of gap lines, and line by line the decision takes
minutes. This strategy drops most pairs of lines at once by the number of common words regardless
of order, counts the rest together and fills every row of the table at once.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from iec.answer_validator.alignment.aligned_answer import PAIRING_GLUE, PAIRING_ONE, PAIRING_SPLIT
from iec.answer_validator.alignment.gaps.gap_decision import GapDecision, Move, decide_from_table
from iec.answer_validator.alignment.gaps.word_matrices import (
    encode_lines,
    lcs_many,
    position_matches,
    shared_counts,
    word_masks,
)

#: The longest reference line of a gap this strategy decides: a glue of two lines must fit the
#: bits of a machine word.
LONGEST_MASKED_LINE = 32


@dataclass(frozen=True, slots=True)
class PairCounts:
    """The pairings that passed the threshold: for a row of the table - columns and counts.

    ``rows`` keeps them as arrays for filling the table, ``lists`` as lists for looking up one cell.
    """

    rows: dict[int, tuple[np.ndarray, np.ndarray]]
    lists: dict[int, tuple[list[int], list[int]]]

    @classmethod
    def of(cls, rows: np.ndarray, columns: np.ndarray, counts: np.ndarray) -> PairCounts:
        """Group the pairings by rows of the table, the columns sorted."""
        order = np.lexsort((columns, rows))
        rows, columns, counts = rows[order], columns[order], counts[order]
        borders = np.flatnonzero(np.diff(rows)) + 1
        grouped = {
            int(part_rows[0]): (part_columns, part_counts)
            for part_rows, part_columns, part_counts in zip(
                np.split(rows, borders),
                np.split(columns, borders),
                np.split(counts, borders),
                strict=True,
            )
            if len(part_rows)
        }
        return cls(
            grouped,
            {
                row: (part_columns.tolist(), part_counts.tolist())
                for row, (part_columns, part_counts) in grouped.items()
            },
        )

    def count(self, row: int, column: int) -> int | None:
        """Return the matched words of a pairing, ``None`` if it did not pass the threshold."""
        found = self.lists.get(row)
        if found is None:
            return None
        columns, counts = found
        place = bisect_left(columns, column)
        if place == len(columns) or columns[place] != column:
            return None
        return counts[place]


class LargeGapResolver:
    """The strategy of a large gap of short lines: the table filled by whole matrices.

    It decides a gap whose reference lines are at most :data:`LONGEST_MASKED_LINE` words long and
    gives the same decision as the small strategy.
    """

    def resolve(
        self,
        gap_reference: Sequence[int],
        gap_answer: Sequence[int],
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        copies: frozenset[tuple[str, ...]],
    ) -> GapDecision:
        """Decide a gap by matrices; the arguments and the result are those of ``GapResolver``."""
        return _resolve_large_gap(
            gap_reference, gap_answer, reference_signatures, answer_signatures, copies
        )


def _resolve_large_gap(
    gap_reference: Sequence[int],
    gap_answer: Sequence[int],
    reference_signatures: Sequence[tuple[str, ...]],
    answer_signatures: Sequence[tuple[str, ...]],
    copies: frozenset[tuple[str, ...]],
) -> GapDecision:
    """Decide a gap by whole matrices: the same table and the same walk as on a small gap.

    The number of common words regardless of order drops most pairs of lines at once. The pairs
    that remain are counted together, and the rows of the table are filled at once from them.
    """
    reference = list(gap_reference)
    copied = [j for j in gap_answer if answer_signatures[j] in copies]
    candidates = [j for j in gap_answer if answer_signatures[j] not in copies]
    if not candidates:
        return decide_from_table(
            reference,
            [],
            copied,
            lambda a, b: 0,
            lambda a, b: [],
            reference_signatures,
            answer_signatures,
        )
    ids: dict[str, int] = {}
    lines = encode_lines([reference_signatures[i] for i in reference], ids)
    answer = encode_lines([answer_signatures[j] for j in candidates], ids)
    masks = word_masks(lines)
    shared = shared_counts(answer, lines)
    alive = np.zeros(len(candidates), dtype=bool)

    # One to one: answer line k against reference line i.
    shorter = np.minimum.outer(answer.sizes, lines.sizes)
    k_one, i_one = np.nonzero(2 * shared > shorter)
    same = answer.sizes[k_one] == lines.sizes[i_one]
    count_one = np.zeros(len(k_one), dtype=np.int64)
    count_one[same] = position_matches(answer, lines, k_one[same], i_one[same])
    count_one[~same] = lcs_many(
        answer.flat,
        answer.starts[k_one[~same]],
        answer.sizes[k_one[~same]],
        masks,
        i_one[~same],
    )
    valid = 2 * count_one > shorter[k_one, i_one]
    k_one, i_one, count_one = k_one[valid], i_one[valid], count_one[valid]
    alive[k_one] = True

    # A glue: answer line k against reference lines t and t + 1.
    glued = lines.sizes[:-1] + lines.sizes[1:]
    shorter = np.minimum.outer(answer.sizes, glued)
    k_glue, t_glue = np.nonzero(2 * (shared[:, :-1] + shared[:, 1:]) > shorter)
    count_glue = lcs_many(
        answer.flat,
        answer.starts[k_glue],
        answer.sizes[k_glue],
        masks,
        t_glue,
        lines.sizes[t_glue],
    )
    valid = 2 * count_glue > shorter[k_glue, t_glue]
    k_glue, t_glue, count_glue = k_glue[valid], t_glue[valid], count_glue[valid]
    alive[k_glue] = True

    # A split: neighbouring answer lines k and k + 1 against reference line i.
    numbers = np.array(candidates, dtype=np.int64)
    firsts = np.flatnonzero(numbers[1:] == numbers[:-1] + 1)
    split_sizes = answer.sizes[firsts] + answer.sizes[firsts + 1]
    shorter = np.minimum.outer(split_sizes, lines.sizes)
    pair_split, i_split = np.nonzero(2 * (shared[firsts] + shared[firsts + 1]) > shorter)
    count_split = lcs_many(
        answer.flat,
        answer.starts[firsts[pair_split]],
        split_sizes[pair_split],
        masks,
        i_split,
    )
    valid = 2 * count_split > shorter[pair_split, i_split]
    k_split, i_split, count_split = firsts[pair_split[valid]], i_split[valid], count_split[valid]
    alive[k_split] = True
    alive[k_split + 1] = True
    del shared, shorter, masks

    # Column of the table of every candidate that survived: from one, zero is no line yet.
    column = np.cumsum(alive)
    survivors = [candidates[k] for k in np.flatnonzero(alive).tolist()]
    lonely = [candidates[k] for k in np.flatnonzero(~alive).tolist()]
    one = PairCounts.of(i_one + 1, column[k_one], count_one)
    split = PairCounts.of(i_split + 1, column[k_split + 1], count_split)
    glue = PairCounts.of(t_glue + 2, column[k_glue], count_glue)

    best = np.zeros((len(reference) + 1, len(survivors) + 1), dtype=np.int32)
    for a in range(1, len(reference) + 1):
        row = best[a - 1].copy()
        pairs = one.rows.get(a)
        if pairs is not None:
            row[pairs[0]] = np.maximum(row[pairs[0]], best[a - 1, pairs[0] - 1] + 2 * pairs[1])
        pairs = split.rows.get(a)
        if pairs is not None:
            row[pairs[0]] = np.maximum(row[pairs[0]], best[a - 1, pairs[0] - 2] + 2 * pairs[1] - 3)
        pairs = glue.rows.get(a)
        if pairs is not None:
            row[pairs[0]] = np.maximum(row[pairs[0]], best[a - 2, pairs[0] - 1] + 2 * pairs[1] - 3)
        best[a] = np.maximum.accumulate(row)

    def moves(a: int, b: int) -> list[Move]:
        """Return the pairings that end at the cell ``(a, b)`` of the table."""
        found: list[Move] = []
        if a >= 1 and b >= 1:
            count = one.count(a, b)
            if count is not None:
                found.append((PAIRING_ONE, 1, 1, count))
        if a >= 1 and b >= 2:
            count = split.count(a, b)
            if count is not None:
                found.append((PAIRING_SPLIT, 1, 2, count))
        if a >= 2 and b >= 1:
            count = glue.count(a, b)
            if count is not None:
                found.append((PAIRING_GLUE, 2, 1, count))
        return found

    return decide_from_table(
        reference,
        survivors,
        (*copied, *lonely),
        lambda a, b: int(best[a, b]),
        moves,
        reference_signatures,
        answer_signatures,
    )
