"""The decision of a gap between anchors: its groups, the copy threshold and the walk back.

A gap is the run of reference lines and answer lines between two anchors. Its decision pairs gap
lines so that as many words as possible match: one answer line against one reference line, one
against two neighbouring reference lines (a glue) or two neighbouring answer lines against one (a
split). A pairing is allowed only if more than half of the words of the shorter side match (the
copy threshold). Both strategies of deciding a gap fill the same table ``best[a][b]`` - the best
value of the first ``a`` reference lines against the first ``b`` paired answer lines - and the
walk back over it, written here once, turns the table into the decision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from iec.answer_validator.alignment.aligned_answer import (
    GROUP_DELETED,
    GROUP_INSERTED,
    PAIRING_ONE,
)

#: Copy threshold: lines are paired if more than this share of the words of the shorter side
#: match. A word of a foreign line matches a reference word by chance with probability about
#: 1/D, a word of a rewritten line almost always.
COPY_SHARE = 0.5

#: A pairing that ends at a cell of the table: its kind, the numbers of reference lines and of
#: answer lines it takes, and its matched words.
Move = tuple[str, int, int, int]
#: The best value of the table at a cell ``(a, b)``.
Cell = Callable[[int, int], int]
#: The pairings that end at a cell ``(a, b)``.
Moves = Callable[[int, int], list[Move]]


@dataclass(frozen=True, slots=True)
class Group:
    """A decision on a part of a gap: the kind and the reference and answer lines in it.

    The kind is a pairing (``one``, ``glue``, ``split``), ``deleted`` (a reference line without a
    pair) or ``inserted`` (an answer line without a pair).
    """

    kind: str
    reference_lines: tuple[int, ...]
    answer_lines: tuple[int, ...]


#: The decision of a gap: its groups in the order of the lines and its pairings, each with the
#: share of matched words of the shorter side.
GapDecision = tuple[list[Group], list[tuple[str, float]]]


def is_copy(matched: int, answer_size: int, reference_size: int) -> bool:
    """Tell whether a pair passes the copy threshold: more than the share of the shorter side."""
    return matched > COPY_SHARE * min(answer_size, reference_size)


def pairing_value(kind: str, count: int) -> int:
    """Return the value of a pairing in half words.

    The value is ``2·count``, three less for a glue and a split: these are taken only if they give
    at least two matched words more than one to one, because one extra match can be chance.
    """
    return 2 * count - (0 if kind == PAIRING_ONE else 3)


def decide_from_table(
    reference: Sequence[int],
    survivors: Sequence[int],
    unpaired: Sequence[int],
    best: Cell,
    moves: Moves,
    reference_signatures: Sequence[tuple[str, ...]],
    answer_signatures: Sequence[tuple[str, ...]],
) -> GapDecision:
    """Walk the table back from its end and return the decisions of the gap and its pairings.

    Args:
        reference: The reference lines of the gap, in a row.
        survivors: The answer lines that pass the copy threshold in some pairing, in order.
        unpaired: The answer lines that are insertions: copies of reference lines and lines that
            pass the threshold in no pairing.
        best: The best value of the table at a cell.
        moves: The pairings that end at a cell.
        reference_signatures: The canonical form of the reference lines.
        answer_signatures: The canonical form of the answer lines.

    Returns:
        The groups of the gap in the order of the lines followed by the unpaired answer lines,
        and the pairings with the share of matched words of the shorter side.
    """
    groups: list[Group] = []
    pairings: list[tuple[str, float]] = []
    a, b = len(reference), len(survivors)
    while a or b:
        for kind, step_a, step_b, count in moves(a, b):
            if best(a - step_a, b - step_b) + pairing_value(kind, count) == best(a, b):
                answer_part = tuple(survivors[b - step_b : b])
                reference_part = tuple(reference[a - step_a : a])
                groups.append(Group(kind, reference_part, answer_part))
                shorter = min(
                    sum(len(answer_signatures[j]) for j in answer_part),
                    sum(len(reference_signatures[i]) for i in reference_part),
                )
                pairings.append((kind, count / shorter))
                a, b = a - step_a, b - step_b
                break
        else:
            if a and best(a - 1, b) == best(a, b):
                groups.append(Group(GROUP_DELETED, (reference[a - 1],), ()))
                a -= 1
            else:
                groups.append(Group(GROUP_INSERTED, (), (survivors[b - 1],)))
                b -= 1
    groups.reverse()
    pairings.reverse()
    groups.extend(Group(GROUP_INSERTED, (), (j,)) for j in unpaired)
    return groups, pairings
