"""The longest chain of candidate pairs increasing in both lines: the anchors of an answer.

It is the longest common subsequence of the lines of the text and of the answer, with the equality of
lines judged by the recognizers, found by the patience method (patience sorting) in time O(P log P)
for P pairs: ``tails[k]`` is the smallest last text line of a chain of k + 1 pairs found so far.
"""

from __future__ import annotations

import numpy as np

from iec.infrastructure.acceleration import compiled


def longest_chain(answer_lines: np.ndarray, text_lines: np.ndarray) -> list[tuple[int, int]]:
    """Return the anchors: the pairs (text line, answer line) of the longest chain increasing in both.

    Args:
        answer_lines: The answer lines of the candidate pairs, increasing.
        text_lines: Their text lines; within one answer line decreasing, so that one answer line
            never stands twice in a chain.

    Returns:
        The pairs of the chain in order. A tail is replaced only by a strictly smaller text line, so
        of chains of equal length the one with the earlier answer lines is kept.
    """
    chosen = _chain(text_lines)
    return list(zip(text_lines[chosen].tolist(), answer_lines[chosen].tolist(), strict=True))


@compiled
def _chain(text_lines: np.ndarray) -> np.ndarray:
    """Return the numbers of the candidate pairs of the longest increasing chain, in order."""
    count = len(text_lines)
    tails = np.empty(count, np.int64)
    ends = np.empty(count, np.int64)
    parents = np.empty(count, np.int64)
    length = 0
    for number in range(count):
        text = text_lines[number]
        low = 0
        high = length
        while low < high:
            middle = (low + high) // 2
            if tails[middle] < text:
                low = middle + 1
            else:
                high = middle
        parents[number] = ends[low - 1] if low > 0 else -1
        if low == length:
            tails[low] = text
            ends[low] = number
            length += 1
        elif text < tails[low]:
            tails[low] = text
            ends[low] = number
    chain = np.empty(length, np.int64)
    number = ends[length - 1] if length > 0 else -1
    place = length - 1
    while number >= 0:
        chain[place] = number
        place -= 1
        number = parents[number]
    return chain
