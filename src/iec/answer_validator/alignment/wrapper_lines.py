"""The wrapper lines of an answer: lines without a single task word at its edges.

A wrapper line stands at an edge of the answer and has no task word, such as a line of introduction
before the text; it is no place of the alignment.
"""

from __future__ import annotations

from collections.abc import Sequence


def wrapper_lines(
    answer_lines: Sequence[tuple[str, ...]],
    anchors: Sequence[tuple[int, int]],
    task_words: frozenset[str],
) -> frozenset[int]:
    """Return the wrapper lines: lines at the edges of the answer without a single task word.

    An edge is the start of the answer before the first anchor and the end after the last one;
    wrapper lines run in a row from the very edge. An empty line has no task words. A line with
    at least one task word is never a wrapper: it is rewritten text with errors, and it goes into
    the alignment.
    """
    first = anchors[0][1] if anchors else len(answer_lines)
    last = anchors[-1][1] if anchors else -1
    wrapper: set[int] = set()
    index = 0
    while index < first and not task_words.intersection(answer_lines[index]):
        wrapper.add(index)
        index += 1
    index = len(answer_lines) - 1
    floor = max(last, max(wrapper, default=-1))
    while index > floor and not task_words.intersection(answer_lines[index]):
        wrapper.add(index)
        index -= 1
    return frozenset(wrapper)
