"""The strategies that recognize the text lines an answer line can stand on.

Three kinds of evidence, from the strongest. The text lines equal to the answer line. Otherwise the text
lines with the same skeleton where more than half of the places agree: the line differs from them only
by left-hand words of the rules, as after a wrong replacement. Otherwise the text lines of the same
length where every marker of the answer line - a background word of the text on its place - stands on
its place too and more than half of the places agree, as after a misfire, which puts the word of a rule
on the place of a background word. The finder asks the strategies in this order, and an answer line
recognized by one is not asked further. Within one answer line the candidates go by decreasing text
lines, as the longest chain needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from iec.answer_validator.alignment.anchors.text_line_index import AnswerLines, TextLineIndex
from iec.infrastructure.acceleration import compiled


@dataclass(frozen=True, slots=True)
class Candidates:
    """Pairs (answer line, text line): the text lines an answer line can stand on.

    By increasing answer lines; within one answer line, by decreasing text lines.
    """

    answer_lines: np.ndarray
    text_lines: np.ndarray


class LineRecognizer(Protocol):
    """A strategy that finds the text lines the open answer lines can stand on."""

    def recognize(
        self, index: TextLineIndex, answer: AnswerLines, open_lines: np.ndarray
    ) -> Candidates:
        """Return the candidates of the open answer lines.

        Args:
            index: The index of the text lines.
            answer: The answer lines as numbers of the index.
            open_lines: The answer lines no stronger strategy recognized, increasing.

        Returns:
            The candidates; an answer line without one is left out.
        """
        ...


class EqualLineRecognizer:
    """The text lines equal to the answer line."""

    def recognize(
        self, index: TextLineIndex, answer: AnswerLines, open_lines: np.ndarray
    ) -> Candidates:
        """Return the equal text lines; the arguments and the result are those of ``LineRecognizer``."""
        hashes = answer.hashes[open_lines]
        lines, texts = _hashed_pairs(
            open_lines,
            np.searchsorted(index.line_hashes, hashes, side="left"),
            np.searchsorted(index.line_hashes, hashes, side="right"),
            index.line_order,
            index.matrix,
            index.matrix,
            index.sizes,
            answer.flat,
            answer.flat,
            answer.starts,
            answer.sizes,
            False,
        )
        return Candidates(lines, texts)


class SkeletonRecognizer:
    """The text lines with the same skeleton where more than half of the places agree."""

    def recognize(
        self, index: TextLineIndex, answer: AnswerLines, open_lines: np.ndarray
    ) -> Candidates:
        """Return the text lines of the same skeleton; arguments and result of ``LineRecognizer``."""
        hashes = answer.skeleton_hashes[open_lines]
        lines, texts = _hashed_pairs(
            open_lines,
            np.searchsorted(index.skeleton_hashes, hashes, side="left"),
            np.searchsorted(index.skeleton_hashes, hashes, side="right"),
            index.skeleton_order,
            index.skeletons,
            index.matrix,
            index.sizes,
            answer.skeleton,
            answer.flat,
            answer.starts,
            answer.sizes,
            True,
        )
        return Candidates(lines, texts)


class MarkerRecognizer:
    """The text lines of the same length that keep every marker of the answer line and most places."""

    def recognize(
        self, index: TextLineIndex, answer: AnswerLines, open_lines: np.ndarray
    ) -> Candidates:
        """Return the text lines that keep the markers; arguments and result of ``LineRecognizer``."""
        lines, texts = _marker_pairs(
            open_lines,
            answer.flat,
            answer.starts,
            answer.sizes,
            index.rule_word,
            index.text_word,
            index.matrix,
            index.sizes,
            index.place_order,
            index.place_words,
        )
        return Candidates(lines, texts)


@compiled
def _hashed_pairs(
    open_lines: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    order: np.ndarray,
    compare: np.ndarray,
    values: np.ndarray,
    sizes: np.ndarray,
    answer_compare: np.ndarray,
    answer_values: np.ndarray,
    starts: np.ndarray,
    line_sizes: np.ndarray,
    majority: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the text lines of equal hash of every open line word by word.

    ``low``/``high`` bound the lines of equal hash in ``order``, where equal hashes go by increasing
    line, so the walk from ``high`` down gives the text lines of an answer line decreasing. A pair stays
    when the rows ``compare`` and ``answer_compare`` are equal and, with ``majority``, more than half
    of the places of ``values`` and ``answer_values`` agree.
    """
    total = 0
    for k in range(len(open_lines)):
        total += high[k] - low[k]
    lines = np.empty(total, np.int64)
    texts = np.empty(total, np.int64)
    found = 0
    for k in range(len(open_lines)):
        line = open_lines[k]
        start = starts[line]
        size = line_sizes[line]
        for position in range(high[k] - 1, low[k] - 1, -1):
            text = order[position]
            if sizes[text] != size:
                continue
            same = True
            for place in range(size):
                if compare[text, place] != answer_compare[start + place]:
                    same = False
                    break
            if not same:
                continue
            if majority:
                agree = 0
                for place in range(size):
                    if values[text, place] == answer_values[start + place]:
                        agree += 1
                if 2 * agree <= size:
                    continue
            lines[found] = line
            texts[found] = text
            found += 1
    return lines[:found], texts[:found]


@compiled
def _bound(column: np.ndarray, value: int, upper: bool) -> int:
    """Return the first place of the sorted ``column`` with a word greater than ``value`` (``upper``)
    or not less than it: the standard binary search of the upper and the lower bound."""
    low = 0
    high = len(column)
    while low < high:
        middle = (low + high) // 2
        if column[middle] < value or (upper and column[middle] == value):
            low = middle + 1
        else:
            high = middle
    return low


@compiled
def _marker_pairs(
    open_lines: np.ndarray,
    flat: np.ndarray,
    starts: np.ndarray,
    line_sizes: np.ndarray,
    rule_word: np.ndarray,
    text_word: np.ndarray,
    matrix: np.ndarray,
    sizes: np.ndarray,
    place_order: np.ndarray,
    place_words: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The text lines that keep every marker of an open line and more than half of its places.

    A marker is a word of the line that occurs in the text and is no left-hand word of a rule. The
    candidates of a line are the text lines with its rarest marker on the same place; a marker that
    no text line has on its place leaves the line without candidates. The first pass finds the rarest
    marker of every line, the second checks its candidates.
    """
    count = len(open_lines)
    best_place = np.full(count, -1, np.int64)
    best_low = np.zeros(count, np.int64)
    best_high = np.zeros(count, np.int64)
    total = 0
    for k in range(count):
        line = open_lines[k]
        start = starts[line]
        empty = False
        for place in range(line_sizes[line]):
            word = flat[start + place]
            if word < 0 or not text_word[word] or rule_word[word]:
                continue
            low = _bound(place_words[place], word, False)
            high = _bound(place_words[place], word, True)
            if high == low:
                empty = True
                break
            if best_place[k] < 0 or high - low < best_high[k] - best_low[k]:
                best_place[k] = place
                best_low[k] = low
                best_high[k] = high
        if empty:
            best_place[k] = -1
        if best_place[k] >= 0:
            total += best_high[k] - best_low[k]
    lines = np.empty(total, np.int64)
    texts = np.empty(total, np.int64)
    found = 0
    for k in range(count):
        chosen = best_place[k]
        if chosen < 0:
            continue
        line = open_lines[k]
        start = starts[line]
        size = line_sizes[line]
        for position in range(best_high[k] - 1, best_low[k] - 1, -1):
            text = place_order[chosen, position]
            if sizes[text] != size:
                continue
            keeps = True
            agree = 0
            for place in range(size):
                word = flat[start + place]
                if matrix[text, place] == word:
                    agree += 1
                elif word >= 0 and text_word[word] and not rule_word[word]:
                    keeps = False
                    break
            if keeps and 2 * agree > size:
                lines[found] = line
                texts[found] = text
                found += 1
    return lines[:found], texts[:found]
