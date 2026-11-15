"""Mirror tests of the anchors of lines."""

from iec.answer_validator.alignment.line_anchors import anchor_lines


def test_anchor_lines_prefer_the_first_pass() -> None:
    pairs = anchor_lines([("a",), ("b",), ("c",)], [("a",), ("b",), ("a",), ("b",), ("c",)])

    assert pairs == [(0, 0), (1, 1), (2, 4)]


def test_equal_lines_stand_on_the_diagonal() -> None:
    lines = [("a",), ("b",), ("c",)]
    assert anchor_lines(lines, lines) == [(0, 0), (1, 1), (2, 2)]


def test_an_equal_start_keeps_the_first_pass_against_the_reference() -> None:
    pairs = anchor_lines([("a",), ("b",), ("c",)], [("a",), ("b",), ("c",), ("b",), ("c",)])
    assert pairs == [(0, 0), (1, 1), (2, 2)]


def test_anchors_after_an_equal_start_keep_their_numbers() -> None:
    pairs = anchor_lines([("a",), ("b",), ("c",), ("d",)], [("a",), ("x",), ("c",), ("d",)])
    assert pairs == [(0, 0), (2, 2), (3, 3)]
