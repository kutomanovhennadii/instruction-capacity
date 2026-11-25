"""Mirror tests of the longest chain."""

from __future__ import annotations

import numpy as np

from iec.answer_validator.alignment.anchors import longest_chain


def _pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    answer = np.array([a for a, _ in pairs], dtype=np.int64)
    text = np.array([t for _, t in pairs], dtype=np.int64)
    return longest_chain(answer, text)


def test_the_chain_passes_a_far_pair() -> None:
    assert _pairs([(0, 0), (1, 1), (2, 9), (3, 2), (4, 3)]) == [(0, 0), (1, 1), (2, 3), (3, 4)]


def test_of_equal_chains_the_earlier_answer_lines_stay() -> None:
    assert _pairs([(0, 0), (1, 1), (2, 0), (3, 1), (4, 2)]) == [(0, 0), (1, 1), (2, 4)]


def test_no_pairs_give_no_chain() -> None:
    assert _pairs([]) == []
