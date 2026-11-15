"""The text forms of an answer: the answer without trailing whitespace and its lines of words."""

from __future__ import annotations


def normalize_output(text: str) -> str:
    """Strip trailing whitespace from the model answer - and nothing more.

    A line break at the end of generation carries no information about replacements. Case is not
    folded: vocabulary words are lowercase, and ``Stone`` is a wrong word.
    """
    return text.rstrip()


def split_lines(text: str) -> tuple[tuple[str, ...], ...]:
    """Cut the text into lines, and the lines into words by whitespace.

    An empty line is kept as a line without words: otherwise an inserted line break would vanish
    without a trace.
    """
    return tuple(tuple(line.split()) for line in text.split("\n"))
