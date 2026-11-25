"""The anchors of an answer by the background words of its lines.

An answer line is recognized by the strongest evidence that exists - an equal text line, the same
skeleton, the same markers (``line_recognizers``) - and of all recognized pairs the anchors are the
longest chain that goes forward both in the text and in the answer (``longest_chain``): the longest
common subsequence of the standard comparison of texts, with the equality of lines judged by the
recognizers. A line that matches only text lines already passed falls into a gap, and a single line
from a far place of the text cannot carry the anchors away: a chain through it is shorter.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from iec.answer_validator.alignment.anchors.anchor_finder import AnchorFinder
from iec.answer_validator.alignment.anchors.line_recognizers import (
    EqualLineRecognizer,
    LineRecognizer,
    MarkerRecognizer,
    SkeletonRecognizer,
)
from iec.answer_validator.alignment.anchors.longest_chain import longest_chain
from iec.answer_validator.alignment.anchors.text_line_index import (
    build_text_line_index,
    encode_answer_lines,
)


class MarkerAnchorFinder:
    """The anchors by background words: the recognizers in their order, then the longest chain.

    The strategy :class:`AnchorFinder` of the validator by default. Its recognizers come through the
    constructor, from the strongest evidence; an answer line recognized by one is not asked further.
    """

    def __init__(self, recognizers: Sequence[LineRecognizer]) -> None:
        """Remember the recognizers.

        Args:
            recognizers: The recognizers, from the strongest evidence to the weakest.
        """
        self._recognizers = tuple(recognizers)

    def find(
        self,
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        sources: frozenset[str],
    ) -> list[tuple[int, int]]:
        """Return the anchors; the arguments and the result are those of ``AnchorFinder``."""
        if not reference_signatures or not answer_signatures:
            return []
        index = build_text_line_index(reference_signatures, sources)
        if index.matrix.size == 0:
            return []
        answer = encode_answer_lines(index, answer_signatures)
        open_lines = np.flatnonzero(np.isin(answer.sizes, index.sizes))
        answer_parts: list[np.ndarray] = []
        text_parts: list[np.ndarray] = []
        for recognizer in self._recognizers:
            if len(open_lines) == 0:
                break
            found = recognizer.recognize(index, answer, open_lines)
            answer_parts.append(found.answer_lines)
            text_parts.append(found.text_lines)
            open_lines = np.setdiff1d(open_lines, found.answer_lines)
        if not answer_parts:
            return []
        answer_lines = np.concatenate(answer_parts)
        text_lines = np.concatenate(text_parts)
        order = np.lexsort((-text_lines, answer_lines))
        return longest_chain(answer_lines[order], text_lines[order])


#: The anchor finder of the validator: equal lines, then the same skeleton, then the markers.
DEFAULT_ANCHOR_FINDER: AnchorFinder = MarkerAnchorFinder(
    (EqualLineRecognizer(), SkeletonRecognizer(), MarkerRecognizer())
)
