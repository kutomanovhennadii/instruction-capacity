"""The alignment of the words of paired lines with the fewest edits."""

from __future__ import annotations

from collections.abc import Sequence


def edit_alignment(
    answer_words: Sequence[str], reference_words: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Return the alignment of words with the fewest edits.

    A pair ``(i, j)`` - answer word ``i`` against reference word ``j`` (matched or replaced);
    ``(None, j)`` - a deletion of a reference word; ``(i, None)`` - an insertion of an answer
    word. A replacement, a deletion and an insertion cost one edit each. Of alignments with equal
    edits the one with more matched words is taken, and at full equality a pair comes first,
    then a deletion, then an insertion.
    """
    rows, columns = len(answer_words), len(reference_words)
    cost = [[0] * (columns + 1) for _ in range(rows + 1)]
    gain = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i
    for j in range(1, columns + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            same = int(answer_words[i - 1] == reference_words[j - 1])
            best_cost, negative_gain = min(
                (cost[i - 1][j - 1] + 1 - same, -(gain[i - 1][j - 1] + same)),
                (cost[i][j - 1] + 1, -gain[i][j - 1]),
                (cost[i - 1][j] + 1, -gain[i - 1][j]),
            )
            cost[i][j] = best_cost
            gain[i][j] = -negative_gain
    pairs: list[tuple[int | None, int | None]] = []
    i, j = rows, columns
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = int(answer_words[i - 1] == reference_words[j - 1])
            if (
                cost[i][j] == cost[i - 1][j - 1] + 1 - same
                and gain[i][j] == gain[i - 1][j - 1] + same
            ):
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if j > 0 and cost[i][j] == cost[i][j - 1] + 1 and gain[i][j] == gain[i][j - 1]:
            pairs.append((None, j - 1))
            j -= 1
            continue
        pairs.append((i - 1, None))
        i -= 1
    pairs.reverse()
    return pairs
