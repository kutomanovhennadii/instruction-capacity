"""Points of a published measurement with their trials: the input of the passport.

A published record keeps what was sent to the model and what it answered. The trials are
not stored: they are rebuilt here by the validator from the record, the same way they were
built when the point was measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from iec.answer_validator import RunValidator, build_reference
from iec.contracts import PointRecord, PointStatus, RawOutput, TrialRecord
from iec.infrastructure.exceptions import ContractViolationError
from iec.point_execution import task_from_record


@dataclass(frozen=True, slots=True)
class MeasuredPoint:
    """A published point together with what the validator read from its answer.

    ``trials`` are the Bernoulli trials of an ``ok`` point, empty for ``na`` and
    ``api_error``; ``first_fail_kind_3_line`` is the answer line where the model lost the
    text (a failure of the third kind), or ``None``.
    """

    record: PointRecord
    trials: tuple[TrialRecord, ...]
    first_fail_kind_3_line: int | None

    @property
    def key(self) -> str:
        """Return the identity of the point as one string: its cluster label in the fit."""
        return "/".join(str(value) for value in self.record.conditions.identity)

    @property
    def D(self) -> int:  # noqa: N802 - protocol axis name
        """Return the number of distinct words of the source text."""
        return self.record.conditions.D

    @property
    def n(self) -> int:
        """Return how many times each word occurs in the source text."""
        return self.record.conditions.n

    @property
    def m(self) -> int:
        """Return the number of rules."""
        return self.record.conditions.m

    @property
    def status(self) -> PointStatus:
        """Return the status of the point."""
        return self.record.status


def measure_points(records: Sequence[PointRecord], vocab_id: str) -> tuple[MeasuredPoint, ...]:
    """Validate the answer of every ``ok`` record; the order of the records is kept.

    Raises:
        ContractViolationError: ``POINT_RECORD_INVALID`` - an ``ok`` record has no answer.
    """
    validator = RunValidator()
    points: list[MeasuredPoint] = []
    for record in records:
        if record.status is not PointStatus.OK:
            points.append(MeasuredPoint(record=record, trials=(), first_fail_kind_3_line=None))
            continue
        if record.answer is None:
            raise ContractViolationError(
                f"point {record.conditions.identity} has status ok but no answer",
                error_code="POINT_RECORD_INVALID",
            )
        table, text = task_from_record(record, vocab_id)
        reference = build_reference(table, text)
        result = validator.validate(RawOutput(text=record.answer), reference, text, table)
        points.append(
            MeasuredPoint(
                record=record,
                trials=tuple(result.records),
                first_fail_kind_3_line=result.first_fail_kind_3_line,
            )
        )
    return tuple(points)
