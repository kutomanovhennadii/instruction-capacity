"""Mirror tests of the wrapper lines."""

from __future__ import annotations

from iec.answer_validator.alignment.wrapper_lines import wrapper_lines

TASK = frozenset({"a", "b", "c"})


def test_lines_without_task_words_at_the_edges_are_the_wrapper() -> None:
    lines = [("Here", "is"), ("a", "b"), ("c",), ("Done",)]
    assert wrapper_lines(lines, [(0, 1), (1, 2)], TASK) == frozenset({0, 3})


def test_a_line_with_a_task_word_is_never_a_wrapper() -> None:
    lines = [("x", "a"), ("a", "b"), ("y",)]
    assert wrapper_lines(lines, [(0, 1)], TASK) == frozenset({2})


def test_without_anchors_the_wrapper_runs_from_both_edges() -> None:
    lines = [("x",), ("a",), ("y",)]
    assert wrapper_lines(lines, [], TASK) == frozenset({0, 2})
