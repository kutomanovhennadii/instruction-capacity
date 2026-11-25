"""Mirror tests of the anchors by background words."""

from __future__ import annotations

import numpy as np

from iec.answer_validator.alignment.anchors import (
    DEFAULT_ANCHOR_FINDER,
    build_text_line_index,
    encode_answer_lines,
)
from iec.answer_validator.alignment.anchors.line_recognizers import _hashed_pairs, _marker_pairs
from iec.answer_validator.alignment.anchors.longest_chain import _chain
from iec.infrastructure.acceleration import plain

#: The left-hand words of the rules: rule words of the text, and what a wrong replacement or a
#: misfire leaves on the place in the canonical form.
SOURCES = frozenset({"lake", "tree"})


def _line(number: int) -> tuple[str, ...]:
    words = [f"w{number}x{place}" for place in range(10)]
    words[1] = "lake"
    words[6] = "tree"
    return tuple(words)


def _text(lines: int) -> list[tuple[str, ...]]:
    return [_line(number) for number in range(lines)]


def _anchors(
    text: list[tuple[str, ...]],
    answer: list[tuple[str, ...]],
    sources: frozenset[str] = SOURCES,
) -> list[tuple[int, int]]:
    return DEFAULT_ANCHOR_FINDER.find(text, answer, sources)


def test_an_exact_copy_stands_line_by_line() -> None:
    assert _anchors(_text(3), _text(3)) == [(0, 0), (1, 1), (2, 2)]


def test_a_wrong_replacement_does_not_move_a_line() -> None:
    answer = _text(3)
    answer[1] = tuple("tree" if place == 1 else word for place, word in enumerate(answer[1]))
    assert _anchors(_text(3), answer) == [(0, 0), (1, 1), (2, 2)]


def test_a_misfire_does_not_move_a_line() -> None:
    answer = _text(3)
    answer[1] = tuple("lake" if place == 4 else word for place, word in enumerate(answer[1]))
    assert _anchors(_text(3), answer) == [(0, 0), (1, 1), (2, 2)]


def test_a_glued_line_is_no_anchor() -> None:
    glued = _line(1)[:3] + _line(2)[3:]
    assert _anchors(_text(4), [_line(0), glued, _line(3)]) == [(0, 0), (3, 2)]


def test_a_lost_line_is_passed() -> None:
    assert _anchors(_text(3), [_line(0), _line(2)]) == [(0, 0), (2, 1)]


def test_a_repeat_keeps_the_first_pass() -> None:
    answer = [*_text(3), _line(0), _line(1)]
    assert _anchors(_text(3), answer) == [(0, 0), (1, 1), (2, 2)]


def test_a_line_with_half_of_its_markers_is_no_anchor() -> None:
    half = tuple(word if place < 5 else "zz" for place, word in enumerate(_line(1)))
    assert _anchors(_text(3), [_line(0), half, _line(2)]) == [(0, 0), (2, 2)]


def test_a_line_without_background_words_takes_the_line_agreeing_most() -> None:
    text = [("lake", "tree", "lake"), ("tree", "tree", "lake")]
    assert _anchors(text, [("tree", "tree", "lake")]) == [(1, 0)]
    assert _anchors(text, [("lake", "lake", "tree")]) == []


def test_a_stray_line_from_far_ahead_does_not_pull_the_anchors() -> None:
    answer = [*_text(3), _line(7), *_text(10)[3:]]
    expected = [(0, 0), (1, 1), (2, 2), *((line, line + 1) for line in range(3, 10))]
    assert _anchors(_text(10), answer) == expected


def test_a_skip_with_fragments_from_far_ahead_keeps_the_long_run() -> None:
    answer = [*_text(3), _line(9), _line(7), *_text(12)[5:]]
    expected = [(0, 0), (1, 1), (2, 2), *((line, line) for line in range(5, 12))]
    assert _anchors(_text(12), answer) == expected


def test_one_background_word_does_not_carry_a_line_far() -> None:
    text = [tuple(f"r{line}{place}" for place in range(9)) + (f"b{line}",) for line in range(6)]
    sources = frozenset(word for line in text for word in line[:9])
    answer = [text[0], text[1][:8] + ("zz", text[1][9]), text[4][:9] + (text[1][9],)]
    assert _anchors(text, answer, sources) == [(0, 0), (1, 1)]


def test_an_empty_text_or_answer_has_no_anchors() -> None:
    assert _anchors([], _text(2)) == []
    assert _anchors(_text(2), []) == []


def test_a_line_of_another_length_stands_nowhere() -> None:
    longer = (*_line(1), "w1x9")
    assert _anchors(_text(3), [_line(0), longer, _line(2)]) == [(0, 0), (2, 2)]


def test_the_compiled_and_the_plain_loops_give_the_same_candidates() -> None:
    rng = np.random.default_rng(7)
    words = [f"v{number}" for number in range(40)]
    sources = frozenset(words[:6])
    text = [tuple(rng.choice(words, size=10)) for _ in range(30)]
    answer = [
        tuple(word if rng.random() > 0.1 else words[int(rng.integers(6))] for word in line)
        for line in (text[int(rng.integers(30))] for _ in range(40))
    ]
    index = build_text_line_index(text, sources)
    lines = encode_answer_lines(index, answer)
    open_lines = np.flatnonzero(np.isin(lines.sizes, index.sizes))
    hashes = lines.skeleton_hashes[open_lines]
    hashed = (
        open_lines,
        np.searchsorted(index.skeleton_hashes, hashes, side="left"),
        np.searchsorted(index.skeleton_hashes, hashes, side="right"),
        index.skeleton_order,
        index.skeletons,
        index.matrix,
        index.sizes,
        lines.skeleton,
        lines.flat,
        lines.starts,
        lines.sizes,
        True,
    )
    marked = (
        open_lines,
        lines.flat,
        lines.starts,
        lines.sizes,
        index.rule_word,
        index.text_word,
        index.matrix,
        index.sizes,
        index.place_order,
        index.place_words,
    )
    for loop, arguments in ((_hashed_pairs, hashed), (_marker_pairs, marked)):
        fast, slow = loop(*arguments), plain(loop)(*arguments)
        assert all(np.array_equal(a, b) for a, b in zip(fast, slow, strict=True))
    chain_input = np.array([3, 1, 2, 2, 5, 4, 6], dtype=np.int64)
    assert np.array_equal(_chain(chain_input), plain(_chain)(chain_input))
