"""Lines of words as numeric matrices: common words and the subsequence of many pairs at once.

A large gap has thousands of pairs of lines. Its strategy turns the lines into numbers once and counts
all pairs together: the common words regardless of order, the words matching by positions and the
longest common subsequence bit by bit, one machine word per pair (Crochemore, Hyyro). The counts of the
common words and of the subsequences are compiled loops (``iec.infrastructure.acceleration``); without
numba the same counts go by numpy steps, with the same result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np

from iec.infrastructure.acceleration import COMPILED, compiled

#: Pairs of lines counted at once: bounds the memory of a gap with thousands of lines.
PAIR_CHUNK = 200_000


def _set_bits_table() -> np.ndarray:
    """Return the number of set bits of every 16-bit value: ``np.bitwise_count`` needs numpy 2."""
    table = np.zeros(1 << 16, dtype=np.uint8)
    for bit in range(16):
        table[1 << bit : 1 << (bit + 1)] = table[: 1 << bit] + 1
    return table


#: The number of set bits of every 16-bit value.
SET_BITS = _set_bits_table()


@dataclass(frozen=True, slots=True)
class EncodedLines:
    """Lines as numbers: the words in a row, where every line starts and how long it is."""

    flat: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray


def encode_lines(lines: Sequence[tuple[str, ...]], ids: dict[str, int]) -> EncodedLines:
    """Turn lines into numbers; the same word gets the same number in all lines."""
    sizes = np.fromiter(map(len, lines), dtype=np.int32, count=len(lines))
    new_words = set(chain.from_iterable(lines)).difference(ids)
    ids.update(zip(new_words, range(len(ids), len(ids) + len(new_words)), strict=True))
    flat = np.fromiter(
        map(ids.__getitem__, chain.from_iterable(lines)),
        dtype=np.int32,
        count=int(sizes.sum()),
    )
    return EncodedLines(flat, np.r_[0, np.cumsum(sizes)[:-1]], sizes)


def word_counts(lines: EncodedLines) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return for every word of every line, sorted by word: the word, the line, the occurrences."""
    total = len(lines.sizes)
    numbers = np.repeat(np.arange(total), lines.sizes)
    keys, counts = np.unique(lines.flat.astype(np.int64) * total + numbers, return_counts=True)
    return keys // total, keys % total, counts


def shared_counts(answer: EncodedLines, reference: EncodedLines) -> np.ndarray:
    """Return the number of common words regardless of order for all pairs of lines.

    Rows - answer lines, columns - reference lines. A word that occurs ``a`` times in an answer
    line and ``r`` times in a reference line adds ``min(a, r)``. The number for two lines together
    is at most the sum of their numbers. A compiled loop when numba is installed, numpy steps
    otherwise; the numbers are the same.
    """
    answer_words, answer_lines, answer_counts = word_counts(answer)
    reference_words, reference_lines, reference_counts = word_counts(reference)
    rows, columns = len(answer.sizes), len(reference.sizes)
    count = _shared_merge if COMPILED else _shared_steps
    return count(
        answer_words,
        answer_lines,
        answer_counts,
        reference_words,
        reference_lines,
        reference_counts,
        rows,
        columns,
    )


def _shared_steps(
    answer_words: np.ndarray,
    answer_lines: np.ndarray,
    answer_counts: np.ndarray,
    reference_words: np.ndarray,
    reference_lines: np.ndarray,
    reference_counts: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    """The common words by numpy steps, word by word: the path without numba."""
    shared = np.zeros((rows, columns), dtype=np.int32)
    for word in np.intersect1d(answer_words, reference_words):
        left, left_end = np.searchsorted(answer_words, (word, word + 1))
        right, right_end = np.searchsorted(reference_words, (word, word + 1))
        place = np.ix_(answer_lines[left:left_end], reference_lines[right:right_end])
        shared[place] += np.minimum.outer(
            answer_counts[left:left_end], reference_counts[right:right_end]
        ).astype(np.int32)
    return shared


@compiled
def _shared_merge(
    answer_words: np.ndarray,
    answer_lines: np.ndarray,
    answer_counts: np.ndarray,
    reference_words: np.ndarray,
    reference_lines: np.ndarray,
    reference_counts: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    """The common words by one compiled loop: the two sorted lists of words are walked together."""
    shared = np.zeros((rows, columns), np.int32)
    left = 0
    right = 0
    while left < len(answer_words) and right < len(reference_words):
        word = answer_words[left]
        other = reference_words[right]
        if word < other:
            left += 1
        elif word > other:
            right += 1
        else:
            left_end = left
            while left_end < len(answer_words) and answer_words[left_end] == word:
                left_end += 1
            right_end = right
            while right_end < len(reference_words) and reference_words[right_end] == word:
                right_end += 1
            for a in range(left, left_end):
                for r in range(right, right_end):
                    shared[answer_lines[a], reference_lines[r]] += np.int32(
                        min(answer_counts[a], reference_counts[r])
                    )
            left = left_end
            right = right_end
    return shared


def position_matches(
    answer: EncodedLines, reference: EncodedLines, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    """Return for pairs of lines of equal length the number of words matching by positions."""
    result = np.zeros(len(rows), dtype=np.int64)
    sizes = reference.sizes[columns]
    for size in np.unique(sizes):
        chosen = np.flatnonzero(sizes == size)
        offsets = np.arange(size)
        for first in range(0, len(chosen), PAIR_CHUNK):
            part = chosen[first : first + PAIR_CHUNK]
            result[part] = (
                answer.flat[answer.starts[rows[part]][:, None] + offsets]
                == reference.flat[reference.starts[columns[part]][:, None] + offsets]
            ).sum(axis=1)
    return result


def word_masks(lines: EncodedLines) -> np.ndarray:
    """Return for every line and every word the bits of the positions of the word in the line.

    Rows - lines, columns - words; the last column stands for a word of no line, it is empty. The
    numbers of the words of the lines must go from zero without gaps.
    """
    words = int(lines.flat.max(initial=-1)) + 1
    masks = np.zeros((len(lines.sizes), words + 1), dtype=np.uint64)
    numbers = np.repeat(np.arange(len(lines.sizes)), lines.sizes)
    positions = (np.arange(len(lines.flat)) - lines.starts[numbers]).astype(np.uint64)
    np.bitwise_or.at(masks, (numbers, lines.flat), np.left_shift(np.uint64(1), positions))
    return masks


def lcs_many(
    answer_flat: np.ndarray,
    answer_starts: np.ndarray,
    answer_sizes: np.ndarray,
    masks: np.ndarray,
    reference_lines: np.ndarray,
    first_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Return the length of the longest common subsequence for many pairs at once, bit by bit.

    The state ``V`` of a pair is one machine word with all bits set. For every answer word with
    the bits ``M`` of its positions in the reference run ``V' = (V + (V & M)) | (V - (V & M))``,
    and the length is the number of zero bits of ``V`` at the end (Crochemore, Hyyro). A reference
    run is the line ``reference_lines``; with ``first_sizes`` it is that line and the next one,
    and the bits of the next line go after ``first_sizes`` bits of the first. A pair drops out
    when its answer words end. A compiled loop pair by pair when numba is installed, numpy steps
    word by word otherwise; the lengths are the same.
    """
    if not COMPILED:
        return _lcs_steps(
            answer_flat, answer_starts, answer_sizes, masks, reference_lines, first_sizes
        )
    sizes = _NO_SIZES if first_sizes is None else first_sizes
    return _lcs_pairs(
        answer_flat.astype(np.int64, copy=False),
        answer_starts.astype(np.int64, copy=False),
        answer_sizes.astype(np.int64, copy=False),
        masks,
        reference_lines.astype(np.int64, copy=False),
        sizes.astype(np.int64, copy=False),
        first_sizes is not None,
    )


#: The first sizes of a call without a glue: the compiled loop takes arrays of one type.
_NO_SIZES = np.zeros(1, dtype=np.int64)


def _lcs_steps(
    answer_flat: np.ndarray,
    answer_starts: np.ndarray,
    answer_sizes: np.ndarray,
    masks: np.ndarray,
    reference_lines: np.ndarray,
    first_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """The subsequences by numpy steps, word by word for all pairs: the path without numba."""
    top = masks.shape[1] - 1
    result = np.zeros(len(answer_sizes), dtype=np.int64)
    for first in range(0, len(answer_sizes), PAIR_CHUNK):
        chosen = slice(first, first + PAIR_CHUNK)
        starts, ends = answer_starts[chosen], answer_sizes[chosen]
        lines = reference_lines[chosen]
        state = np.full(len(ends), np.iinfo(np.uint64).max, dtype=np.uint64)
        for step in range(int(ends.max())):
            active = np.flatnonzero(ends > step)
            words = np.minimum(answer_flat[starts[active] + step], top)
            found = masks[lines[active], words]
            if first_sizes is not None:
                shifts = first_sizes[chosen][active].astype(np.uint64)
                found = found | np.left_shift(masks[lines[active] + 1, words], shifts)
            current = state[active]
            common = current & found
            state[active] = (current + common) | (current - common)
        zeros = ~state
        result[chosen] = (
            SET_BITS[(zeros & np.uint64(0xFFFF)).astype(np.intp)].astype(np.int64)
            + SET_BITS[((zeros >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.intp)]
            + SET_BITS[((zeros >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.intp)]
            + SET_BITS[(zeros >> np.uint64(48)).astype(np.intp)]
        )
    return result


@compiled
def _lcs_pairs(
    answer_flat: np.ndarray,
    answer_starts: np.ndarray,
    answer_sizes: np.ndarray,
    masks: np.ndarray,
    reference_lines: np.ndarray,
    first_sizes: np.ndarray,
    glued: bool,
) -> np.ndarray:
    """The subsequence of every pair by one compiled loop: the steps of ``_lcs_steps``, pair by pair."""
    top = masks.shape[1] - 1
    result = np.zeros(answer_sizes.shape[0], np.int64)
    one = np.uint64(1)
    for pair in range(answer_sizes.shape[0]):
        state = np.uint64(0xFFFFFFFFFFFFFFFF)
        line = reference_lines[pair]
        start = answer_starts[pair]
        for step in range(answer_sizes[pair]):
            word = min(answer_flat[start + step], top)
            found = masks[line, word]
            if glued:
                found = found | (masks[line + 1, word] << np.uint64(first_sizes[pair]))
            common = state & found
            state = (state + common) | (state - common)
        zeros = ~state
        count = 0
        while zeros:
            zeros &= zeros - one
            count += 1
        result[pair] = count
    return result
