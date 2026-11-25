"""The lines of the source text and of an answer as numbers: the structure of speed of the anchors.

Every word of the task gets a number once: the words of the text first, in sorted order, then the
left-hand words of the rules that the text does not contain. A line becomes a row of numbers, and its
hash is one 64-bit number, h = (sum over p of (x_p + 3)·K_p mod 2^64) xor n·K_n, where x_p is the number
of the word on place p, K_p a fixed odd constant of the place, n the length of the line and K_n a fixed
constant of the length. Equal lines have equal hashes; lines with equal hashes are compared word by
word, so a collision costs time, not a wrong anchor. The skeleton of a line puts SKELETON_MARK on the
places of the left-hand words of the rules and keeps the other words. An answer word that is no word of
the task gets UNKNOWN_WORD and equals no word of the text. The index is built once per text
(specification §2.11); its key is the text with the left-hand words of its rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain, repeat

import numpy as np

#: The number of a place after the end of a short line of the text.
PADDING = -1
#: The number of an answer word that is no word of the task.
UNKNOWN_WORD = -2
#: The number that stands for a left-hand word of a rule in a skeleton.
SKELETON_MARK = -3
#: The seed of the constants of the places of the hash: the hashes are the same in every process.
HASH_SEED = 20261125
#: The constant of the length of a line in its hash.
_LENGTH_KEY = np.uint64(0x9E3779B97F4A7C15)


@dataclass(frozen=True, slots=True)
class TextLineIndex:
    """The lines of the source text as numbers, with their hashes and the lines sorted by every place.

    ``numbers`` - the number of every word of the task; ``rule_word``, ``text_word`` - by number,
    whether the word is a left-hand word of a rule and whether it occurs in the text; ``matrix`` and
    ``skeletons`` - the lines and their skeletons as rows of numbers padded with ``PADDING``;
    ``sizes`` - the lengths of the lines; ``line_order``, ``line_hashes`` - the lines sorted by their
    hash, of equal hashes by line, and the hashes in that order; ``skeleton_order``,
    ``skeleton_hashes`` - the same for the skeletons; ``place_order``, ``place_words`` - for every
    place, a row: the lines sorted by the word on that place, of equal words by line, and those words.
    """

    numbers: dict[str, int]
    rule_word: np.ndarray
    text_word: np.ndarray
    matrix: np.ndarray
    skeletons: np.ndarray
    sizes: np.ndarray
    line_order: np.ndarray
    line_hashes: np.ndarray
    skeleton_order: np.ndarray
    skeleton_hashes: np.ndarray
    place_order: np.ndarray
    place_words: np.ndarray


@dataclass(frozen=True, slots=True)
class AnswerLines:
    """The lines of an answer as numbers of an index.

    ``flat`` and ``skeleton`` - the words of all lines in a row and their skeleton; ``starts`` and
    ``sizes`` - where every line starts in them and how long it is; ``hashes`` and
    ``skeleton_hashes`` - the hash of every line and of its skeleton.
    """

    flat: np.ndarray
    skeleton: np.ndarray
    starts: np.ndarray
    sizes: np.ndarray
    hashes: np.ndarray
    skeleton_hashes: np.ndarray


@lru_cache(maxsize=64)
def place_keys(width: int) -> np.ndarray:
    """Return the odd 64-bit constants of the first ``width`` places of the hash.

    Args:
        width: The number of places.

    Returns:
        ``width`` odd ``uint64`` numbers drawn from :data:`HASH_SEED`; the array is shared, do not
        change it.
    """
    drawn = np.random.default_rng(HASH_SEED).integers(0, 2**62, size=width, dtype=np.int64)
    return drawn.astype(np.uint64) * np.uint64(2) + np.uint64(1)


def build_text_line_index(
    reference_signatures: Sequence[tuple[str, ...]], sources: frozenset[str]
) -> TextLineIndex:
    """Build the index of the lines of the source text.

    Args:
        reference_signatures: The lines of the source text.
        sources: The left-hand words of the rules.

    Returns:
        The index of the text.
    """
    text_words = sorted(set(chain.from_iterable(reference_signatures)))
    numbers = {word: number for number, word in enumerate(text_words)}
    for word in sorted(sources):
        numbers.setdefault(word, len(numbers))
    rule_word = np.zeros(len(numbers), dtype=bool)
    rule_word[
        np.fromiter((numbers[word] for word in sources), dtype=np.int64, count=len(sources))
    ] = True
    text_word = np.zeros(len(numbers), dtype=bool)
    text_word[: len(text_words)] = True
    sizes = np.fromiter(
        map(len, reference_signatures), dtype=np.int64, count=len(reference_signatures)
    )
    width = int(sizes.max(initial=0))
    matrix = np.full((len(sizes), width), PADDING, dtype=np.int64)
    matrix[np.repeat(np.arange(len(sizes)), sizes), _places(sizes)] = np.fromiter(
        map(numbers.__getitem__, chain.from_iterable(reference_signatures)),
        dtype=np.int64,
        count=int(sizes.sum()),
    )
    skeletons = np.where((matrix >= 0) & rule_word[np.maximum(matrix, 0)], SKELETON_MARK, matrix)
    keys = place_keys(width)
    line_hashes = _row_hashes(matrix, sizes, keys)
    skeleton_hashes = _row_hashes(skeletons, sizes, keys)
    line_order = np.argsort(line_hashes, kind="stable")
    skeleton_order = np.argsort(skeleton_hashes, kind="stable")
    place_order = np.ascontiguousarray(np.argsort(matrix, axis=0, kind="stable").T)
    return TextLineIndex(
        numbers=numbers,
        rule_word=rule_word,
        text_word=text_word,
        matrix=matrix,
        skeletons=skeletons,
        sizes=sizes,
        line_order=line_order,
        line_hashes=line_hashes[line_order],
        skeleton_order=skeleton_order,
        skeleton_hashes=skeleton_hashes[skeleton_order],
        place_order=place_order,
        place_words=np.ascontiguousarray(np.take_along_axis(matrix.T, place_order, axis=1)),
    )


def encode_answer_lines(
    index: TextLineIndex, answer_signatures: Sequence[tuple[str, ...]]
) -> AnswerLines:
    """Turn the answer lines into numbers of the index, with the hashes of the lines and skeletons.

    Args:
        index: The index of the text lines.
        answer_signatures: The canonical form of the answer lines.

    Returns:
        The answer lines as numbers. A line longer than the text lines is hashed by its first places
        only: such a line stands on no text line, and its hash is never compared.
    """
    sizes = np.fromiter(map(len, answer_signatures), dtype=np.int64, count=len(answer_signatures))
    flat = np.fromiter(
        map(index.numbers.get, chain.from_iterable(answer_signatures), repeat(UNKNOWN_WORD)),
        dtype=np.int64,
        count=int(sizes.sum()),
    )
    skeleton = np.where((flat >= 0) & index.rule_word[np.maximum(flat, 0)], SKELETON_MARK, flat)
    starts = np.cumsum(sizes) - sizes
    width = index.matrix.shape[1]
    keys = np.append(place_keys(width), np.uint64(0))[np.minimum(_places(sizes), width)]
    return AnswerLines(
        flat=flat,
        skeleton=skeleton,
        starts=starts,
        sizes=sizes,
        hashes=_line_hashes(flat, keys, starts, sizes),
        skeleton_hashes=_line_hashes(skeleton, keys, starts, sizes),
    )


def _places(sizes: np.ndarray) -> np.ndarray:
    """Return the place of every word in its line, for lines of the given lengths in a row."""
    return np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(np.cumsum(sizes) - sizes, sizes)


def _row_hashes(rows: np.ndarray, sizes: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Return the hash of every row of numbers: its places with their keys, and its length."""
    filled = np.arange(rows.shape[1]) < sizes[:, None]
    terms = np.where(filled, (rows.astype(np.uint64) + np.uint64(3)) * keys, np.uint64(0))
    return terms.sum(axis=1, dtype=np.uint64) ^ (sizes.astype(np.uint64) * _LENGTH_KEY)


def _line_hashes(
    numbers: np.ndarray, keys: np.ndarray, starts: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """Return the hash of every line of numbers in a row, the same as ``_row_hashes`` of its row.

    The sum of the terms of a line is the difference of two prefix sums: modulo 2^64 the difference
    of the wrapped sums is the wrapped sum of the line.
    """
    terms = (numbers.astype(np.uint64) + np.uint64(3)) * keys
    totals = np.concatenate((np.zeros(1, dtype=np.uint64), np.cumsum(terms, dtype=np.uint64)))
    return (totals[starts + sizes] - totals[starts]) ^ (sizes.astype(np.uint64) * _LENGTH_KEY)
