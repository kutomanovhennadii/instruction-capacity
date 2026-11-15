"""The passport: from the published points to the dictionary of three quantities.

``P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`` is the way to combine the three quantities: it has
no law and no capacity of its own. Every quantity has its own model, laws, level and capacities
(``quantities.sync_section``, ``quantities.rule_word_section``, ``quantities.background_section``),
with the forms of the laws from the assignment of forms. All answered points enter, with no area of
analysis, because the alignment gives an outcome at every place of an answer.

The points are aligned independently of each other, and the three sections are fitted independently
of each other, so a pool of processes (``calculation_pool``) runs both in parallel. The points come
back in the order of the records, and every section is the same computation in another process: the
passport is the same to the last digit with the pool and without it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from typing import Any

import numpy as np

from iec.answer_validator import align_answer, build_reference
from iec.contracts import MeasurementDescription, PointRecord, PointStatus, is_empty_answer
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.quantities.background_design import BackgroundCells, background_cells
from iec.passport.quantities.background_section import background_section
from iec.passport.quantities.rule_word_section import rule_word_section
from iec.passport.quantities.sync_section import sync_section
from iec.passport.trials.layout.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout
from iec.passport.trials.layout.vpm_cube import (
    VpmCube,
    build_layout,
    cell_balanced_weights,
    point_key,
)
from iec.passport.trials.place_trials import PointTrials, point_trials
from iec.passport.trials.trial_tables import TrialTables, build_trial_tables
from iec.point_execution import task_from_record

#: The formula of the three quantities.
FORMULA = "P_correct = P_sync * [pi * H + (1 - pi) * (1 - F)]"
#: The points of one task of the pool in the alignment: few enough to share the long texts out evenly
#: among the workers, enough to keep the cost of sending a task small.
ALIGNMENT_CHUNK = 8

SectionBuilder = Callable[[Sequence[Any], BinLayout, Mapping[str, float], LawAssignment], dict[str, Any]]

#: The sections of the passport in the order of the dictionary: the quantity, its section and the
#: trials of ``MeasurementTrials`` that it reads.
SECTIONS: tuple[tuple[str, SectionBuilder, str], ...] = (
    ("P_sync", sync_section, "points"),
    ("H", rule_word_section, "points"),
    ("F", background_section, "background"),
)


@dataclass(frozen=True, slots=True, eq=False)
class MeasurementTrials:
    """The trials of the three quantities of a measurement: ``points`` and ``background``, by point."""

    points: tuple[PointTrials, ...]
    background: tuple[BackgroundCells, ...]


def _quiet(message: str) -> None:
    """Do not report the progress: the caller did not ask."""
    del message


def _quiet_worker() -> None:
    """Switch off the INFO lines in a worker process: one line per point would flood the output."""
    logging.disable(logging.INFO)


def calculation_pool(workers: int) -> AbstractContextManager[Executor | None]:
    """Return the pool of processes of the calculation, or no pool for one worker.

    Args:
        workers: The number of worker processes; one or less - the calculation runs here.

    Returns:
        A context manager: a ``ProcessPoolExecutor`` whose workers write no INFO lines, or
        ``nullcontext()``, which gives ``None``.
    """
    if workers > 1:
        return ProcessPoolExecutor(max_workers=workers, initializer=_quiet_worker)
    return nullcontext()


def _answered(records: Sequence[PointRecord]) -> list[PointRecord]:
    """Return the records of the answered points; an answer without a word carries no trial."""
    return [
        record
        for record in records
        if record.status is PointStatus.OK and not is_empty_answer(record)
    ]


def align_point(
    record: PointRecord, vocab_id: str, tables: TrialTables
) -> tuple[PointTrials, BackgroundCells]:
    """Align one answered point and return its trials of the three quantities.

    A task of the pool of processes: a function of the module whose arguments pickle, so a worker
    process can run it.

    Args:
        record: The record of an answered point.
        vocab_id: The vocabulary of the measurement.
        tables: The tables of the trials of the measurement (:func:`build_trial_tables`).

    Raises:
        ContractViolationError: ``POINT_NOT_OK`` - the point has no answer.
    """
    conditions = record.conditions
    key = point_key(record)
    if record.answer is None:
        raise ContractViolationError(
            f"point {key} has status ok but no answer", error_code="POINT_NOT_OK"
        )
    table, text = task_from_record(record, vocab_id)
    reference = build_reference(table, text)
    aligned = align_answer(record.answer, reference, text, table)
    trials = point_trials(
        key,
        conditions.D,
        conditions.n,
        conditions.m,
        aligned,
        reference,
        text,
        table,
        tables=tables,
    )
    cells = background_cells(key, conditions.D, conditions.n, conditions.m, aligned, text, table)
    return trials, cells


def measurement_trials(
    description: MeasurementDescription,
    records: Sequence[PointRecord],
    progress: Callable[[str], None] = _quiet,
    *,
    executor: Executor | None = None,
) -> MeasurementTrials:
    """Align every answered point and return its trials.

    The tables of the trials cover the longest text of the measurement and are built once. With a pool
    of processes the points are aligned in its workers, ``ALIGNMENT_CHUNK`` points per task, and come
    back in the order of the records, so the trials are the same as without the pool.

    Args:
        description: The description of the measurement.
        records: The point records.
        progress: Receives a line every 200 points.
        executor: The pool of processes (:func:`calculation_pool`), or ``None`` to align the points
            here, one after another.

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - there are no answered points.
        ContractViolationError: ``POINT_NOT_OK`` - an answered point has no answer.
    """
    answered = _answered(records)
    if not answered:
        raise PassportError("the measurement has no ok points", error_code="PASSPORT_INPUT_MISSING")
    progress(f"{len(answered)} point(s) to align")
    tables = build_trial_tables(
        max(record.conditions.D * record.conditions.n for record in answered)
    )
    vocab_ids, tables_of = repeat(description.vocab_id), repeat(tables)
    aligned = (
        map(align_point, answered, vocab_ids, tables_of)
        if executor is None
        else executor.map(align_point, answered, vocab_ids, tables_of, chunksize=ALIGNMENT_CHUNK)
    )
    points: list[PointTrials] = []
    background: list[BackgroundCells] = []
    for done, (trials, cells) in enumerate(aligned, start=1):
        points.append(trials)
        background.append(cells)
        if done % 200 == 0:
            progress(f"aligned {done}/{len(answered)}")
    return MeasurementTrials(points=tuple(points), background=tuple(background))


def _layout_of(
    records: Sequence[PointRecord],
) -> tuple[BinLayout, dict[str, float], dict[str, Any]]:
    """Return the layout of the bins and the request weights by the cube ``V × P × M``.

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - there are no answered points.
    """
    answered = _answered(records)
    layout = build_layout(answered)
    weighting = cell_balanced_weights(answered, VpmCube(REFERENCE_TOTAL_WORDS))
    if weighting is None:
        raise PassportError("the measurement has no ok points", error_code="PASSPORT_INPUT_MISSING")
    return layout, weighting[0], weighting[1]


def _plain(value: Any) -> Any:
    """Return a value of a section with plain numbers: the JSON file holds no numpy types."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _started(
    executor: Executor | None, build: SectionBuilder, *arguments: Any
) -> Callable[[], dict[str, Any]]:
    """Start a section in the pool, or keep it to run here; return how to take its result."""
    if executor is None:
        return partial(build, *arguments)
    return executor.submit(build, *arguments).result


def build_passport_data(
    description: MeasurementDescription,
    records: Sequence[PointRecord],
    trials: MeasurementTrials,
    name: str,
    assignment: LawAssignment,
    progress: Callable[[str], None] = _quiet,
    *,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Compute the passport from the trials of a measurement.

    The three sections are independent: with a pool of processes they run at once, each in its
    worker; without it they run here in the order of ``SECTIONS``.

    Args:
        description: The description of the measurement.
        records: The point records: they give the layout and the weights.
        trials: The trials of the three quantities (:func:`measurement_trials`).
        name: The name of the measurement.
        assignment: The forms of the laws of the passport (``config/passport_laws.yaml``).
        progress: Receives a line after every quantity.
        executor: The pool of processes (:func:`calculation_pool`), or ``None``.

    Returns:
        The dictionary of ``passport_data.json`` without its format version: ``model``, ``points``,
        ``formula``, ``weighting``, the sections ``P_sync``, ``H``, ``F``.
    """
    layout, weights, weight_summary = _layout_of(records)
    started = [
        (quantity, _started(executor, build, getattr(trials, source), layout, weights, assignment))
        for quantity, build, source in SECTIONS
    ]
    sections: dict[str, dict[str, Any]] = {}
    for quantity, result in started:
        section = result()
        sections[quantity] = section
        progress(
            f"{quantity}: method {section['fit']['method']}, sigma2 {section['fit']['sigma2']}, "
            f"{section['status']}"
        )
    passport = {
        "model": {"id": description.model.model_id, "experiment_id": name},
        "points": len(trials.points),
        "formula": FORMULA,
        "weighting": weight_summary,
        **sections,
    }
    plain: dict[str, Any] = _plain(passport)
    return plain
