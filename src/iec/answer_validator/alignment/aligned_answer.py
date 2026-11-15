"""The result of the alignment of an answer: the kinds and origins of places and their arrays.

A place is a word of the reference or a word of the answer that has no place in the reference.
The alignment puts a word of the answer against every word of the reference or marks a deletion,
and marks a word of the answer without a place as an insertion. A wrong word on its place is not
an alignment failure but the outcome of the trial of a rule word or a background word. A place is
one position of four numeric arrays of equal length; ``AlignedAnswer.places`` builds the objects
on request. The names of the kinds and of the origins live here, and every step of the alignment
takes them from here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Kind of place: a word of the answer stands against a word of the reference, right or not.
PLACE_MATCHED = "matched"
#: Kind of place: a word of the reference has no word of the answer - a deletion, a cut-off.
PLACE_DELETED = "deleted"
#: Kind of place: a word of the answer has no place in the reference - an insertion, a repeat.
PLACE_INSERTED = "inserted"

#: Pairing of gap lines: one answer line against one reference line.
PAIRING_ONE = "one"
#: Pairing of gap lines: one answer line against two neighbouring reference lines.
PAIRING_GLUE = "glue"
#: Pairing of gap lines: two neighbouring answer lines against one reference line.
PAIRING_SPLIT = "split"
#: Origin of a place: an anchor - an answer line exactly equal to a reference line.
GROUP_ANCHOR = "anchor"
#: Origin of a place: a reference line of a gap left without a pair.
GROUP_DELETED = "deleted"
#: Origin of a place: an answer line of a gap left without a pair.
GROUP_INSERTED = "inserted"

#: Numeric codes of the kinds of places in ``AlignedAnswer.kinds``, in this order.
KIND_NAMES: tuple[str, ...] = (PLACE_MATCHED, PLACE_DELETED, PLACE_INSERTED)
KIND_MATCHED, KIND_DELETED, KIND_INSERTED = 0, 1, 2
#: Numeric codes of the origins of places in ``AlignedAnswer.groups``, in this order.
GROUP_NAMES: tuple[str, ...] = (
    GROUP_ANCHOR,
    PAIRING_ONE,
    PAIRING_GLUE,
    PAIRING_SPLIT,
    GROUP_DELETED,
    GROUP_INSERTED,
)
#: The numeric code of every origin name.
GROUP_CODES = {name: code for code, name in enumerate(GROUP_NAMES)}


@dataclass(frozen=True, slots=True)
class Place:
    """One place of the alignment.

    ``kind`` - :data:`PLACE_MATCHED`, :data:`PLACE_DELETED` or :data:`PLACE_INSERTED`;
    ``reference_position`` - the number of the reference word from zero, ``None`` for an
    insertion; ``answer_position`` - the number of the word in the answer core from zero,
    ``None`` for a deletion; ``word`` - the word of the answer, ``None`` for a deletion;
    ``group`` - where the place comes from: :data:`GROUP_ANCHOR` or the kind of the gap decision
    (a pairing, ``deleted``, ``inserted``).
    """

    kind: str
    reference_position: int | None
    answer_position: int | None
    word: str | None
    group: str


@dataclass(frozen=True, slots=True, eq=False)
class AlignedAnswer:
    """The alignment of one answer.

    Every word of the reference and every word of the answer core is in exactly one place; the
    core is the answer without wrapper lines. A place is one position of four arrays of equal
    length: ``kinds`` - the code of its kind (:data:`KIND_NAMES`); ``reference_positions`` and
    ``answer_positions`` - the numbers of the words from zero, ``-1`` for none; ``groups`` - the
    code of its origin (:data:`GROUP_NAMES`). ``core_words`` - the words of the core in order;
    ``core_line_sizes`` - the number of words in every line of the core; ``answer_words`` - words
    in the core; ``wrapper_lines`` - wrapper lines; ``episodes`` - gaps between anchors with a
    deletion or an insertion (synchronisation failures); ``pairings`` - the pairings of lines in
    the gaps: the kind and the share of matched words of the shorter side.
    """

    kinds: np.ndarray
    reference_positions: np.ndarray
    answer_positions: np.ndarray
    groups: np.ndarray
    core_words: tuple[str, ...]
    core_line_sizes: np.ndarray
    answer_words: int
    wrapper_lines: int
    episodes: int
    pairings: tuple[tuple[str, float], ...]

    @property
    def places(self) -> tuple[Place, ...]:
        """Return the places as objects, in the order of the arrays."""
        return tuple(
            Place(
                KIND_NAMES[kind],
                None if reference < 0 else reference,
                None if answer < 0 else answer,
                None if answer < 0 else self.core_words[answer],
                GROUP_NAMES[group],
            )
            for kind, reference, answer, group in zip(
                self.kinds.tolist(),
                self.reference_positions.tolist(),
                self.answer_positions.tolist(),
                self.groups.tolist(),
                strict=True,
            )
        )
