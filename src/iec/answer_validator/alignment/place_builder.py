"""The accumulator of places: anchors and gap decisions turned into the arrays of the alignment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from iec.answer_validator.alignment.aligned_answer import (
    GROUP_ANCHOR,
    GROUP_CODES,
    GROUP_DELETED,
    GROUP_INSERTED,
    KIND_DELETED,
    KIND_INSERTED,
    KIND_MATCHED,
    PAIRING_ONE,
)
from iec.answer_validator.alignment.edit_alignment import edit_alignment
from iec.answer_validator.alignment.gaps.gap_decision import Group


class PlaceBuilder:
    """Accumulator of places: turns anchors and gap decisions into the arrays of the alignment.

    Places of one kind and one origin with consecutive positions are kept as one run of plain
    numbers, and the arrays are built once at the end: an answer without a failure is one run.
    ``unmatched`` counts the deleted and inserted places so far.
    """

    def __init__(
        self,
        source_lines: Sequence[tuple[str, ...]],
        answer_lines: Sequence[tuple[str, ...]],
        signatures: Sequence[tuple[str, ...]],
        reference_start: Sequence[int],
        answer_start: dict[int, int],
    ) -> None:
        """Remember the lines of the task and of the answer and where their words start.

        Args:
            source_lines: The lines of the source text - the canonical form of the reference.
            answer_lines: The lines of the answer.
            signatures: The canonical form of the answer lines.
            reference_start: The number of the first reference word of every reference line.
            answer_start: The number of the first core word of every answer line of the core.
        """
        self._source_lines = source_lines
        self._answer_lines = answer_lines
        self._signatures = signatures
        self._reference_start = reference_start
        self._answer_start = answer_start
        self._runs: list[tuple[int, int, int, int, int]] = []
        self.unmatched = 0

    def add_anchor(self, reference_index: int, answer_index: int) -> None:
        """An anchor: lines of equal length, the words stand against each other by positions."""
        self._add(
            KIND_MATCHED,
            self._reference_start[reference_index],
            self._answer_start[answer_index],
            len(self._answer_lines[answer_index]),
            GROUP_CODES[GROUP_ANCHOR],
        )

    def add_group(self, group: Group) -> None:
        """A gap decision: by positions in a pair of lines of equal length, otherwise by edits."""
        code = GROUP_CODES[group.kind]
        if group.kind == GROUP_DELETED:
            size = len(self._source_lines[group.reference_lines[0]])
            first = self._reference_start[group.reference_lines[0]]
            self._add(KIND_DELETED, first, -1, size, code)
            self.unmatched += size
            return
        if group.kind == GROUP_INSERTED:
            size = len(self._answer_lines[group.answer_lines[0]])
            self._add(KIND_INSERTED, -1, self._answer_start[group.answer_lines[0]], size, code)
            self.unmatched += size
            return
        reference_words = [word for i in group.reference_lines for word in self._source_lines[i]]
        canonical = [word for j in group.answer_lines for word in self._signatures[j]]
        if group.kind == PAIRING_ONE and len(canonical) == len(reference_words):
            self._add(
                KIND_MATCHED,
                self._reference_start[group.reference_lines[0]],
                self._answer_start[group.answer_lines[0]],
                len(canonical),
                code,
            )
            return
        reference_positions = [
            self._reference_start[i] + k
            for i in group.reference_lines
            for k in range(len(self._source_lines[i]))
        ]
        answer_positions = [
            self._answer_start[j] + k
            for j in group.answer_lines
            for k in range(len(self._answer_lines[j]))
        ]
        for answer_k, reference_k in edit_alignment(canonical, reference_words):
            if answer_k is not None and reference_k is not None:
                self._add(
                    KIND_MATCHED,
                    reference_positions[reference_k],
                    answer_positions[answer_k],
                    1,
                    code,
                )
            elif reference_k is not None:
                self._add(KIND_DELETED, reference_positions[reference_k], -1, 1, code)
                self.unmatched += 1
            elif answer_k is not None:
                self._add(KIND_INSERTED, -1, answer_positions[answer_k], 1, code)
                self.unmatched += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the arrays of the places: kinds, reference and answer positions, origins."""
        if not self._runs:
            empty = np.zeros(0, dtype=np.int64)
            return empty.astype(np.int8), empty, empty, empty.astype(np.int8)
        kinds, references, answers, sizes, codes = np.array(self._runs, dtype=np.int64).T
        offsets = np.arange(int(sizes.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(sizes) - sizes, sizes
        )
        reference_firsts = np.repeat(references, sizes)
        answer_firsts = np.repeat(answers, sizes)
        return (
            np.repeat(kinds, sizes).astype(np.int8),
            np.where(reference_firsts < 0, -1, reference_firsts + offsets),
            np.where(answer_firsts < 0, -1, answer_firsts + offsets),
            np.repeat(codes, sizes).astype(np.int8),
        )

    def _add(self, kind: int, reference: int, answer: int, size: int, code: int) -> None:
        """Add a run of places, joining it to the last one if the positions go on in a row."""
        if self._runs:
            last_kind, last_reference, last_answer, last_size, last_code = self._runs[-1]
            if (
                last_kind == kind
                and last_code == code
                and (reference < 0 or reference == last_reference + last_size)
                and (answer < 0 or answer == last_answer + last_size)
            ):
                self._runs[-1] = (kind, last_reference, last_answer, last_size + size, code)
                return
        self._runs.append((kind, reference, answer, size, code))
