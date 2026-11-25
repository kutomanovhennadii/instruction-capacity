"""Mirror tests of the index of the lines of the text."""

from __future__ import annotations

import numpy as np

from iec.answer_validator.alignment.anchors import build_text_line_index, encode_answer_lines
from iec.answer_validator.alignment.anchors.text_line_index import SKELETON_MARK, UNKNOWN_WORD

TEXT = [("a", "lake", "b"), ("c", "tree", "d"), ("a", "lake", "b")]
SOURCES = frozenset({"lake", "tree", "pond"})


def test_the_words_of_the_text_come_first_then_the_rule_words_it_lacks() -> None:
    index = build_text_line_index(TEXT, SOURCES)
    assert sorted(index.numbers, key=index.numbers.__getitem__) == [
        "a",
        "b",
        "c",
        "d",
        "lake",
        "tree",
        "pond",
    ]
    assert index.text_word.tolist() == [True] * 6 + [False]
    assert index.rule_word.tolist() == [False] * 4 + [True] * 3


def test_equal_lines_have_equal_hashes_and_skeletons_mark_the_rule_words() -> None:
    index = build_text_line_index(TEXT, SOURCES)
    by_line = np.empty(3, dtype=np.uint64)
    by_line[index.line_order] = index.line_hashes
    assert by_line[0] == by_line[2] != by_line[1]
    assert index.skeletons[:, 1].tolist() == [SKELETON_MARK] * 3


def test_an_answer_line_has_the_hash_of_its_text_line_and_unknown_words_equal_nothing() -> None:
    index = build_text_line_index(TEXT, SOURCES)
    answer = encode_answer_lines(index, [*TEXT, ("a", "x", "b")])
    by_line = np.empty(3, dtype=np.uint64)
    by_line[index.line_order] = index.line_hashes
    assert answer.hashes[:3].tolist() == by_line.tolist()
    assert answer.flat[-2] == UNKNOWN_WORD
