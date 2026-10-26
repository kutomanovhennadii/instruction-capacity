"""Design matrix of the passport: one row per trial, and the exclusion rule of bins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from iec.contracts import PointStatus
from iec.passport.bin_layout import BinLayout
from iec.passport.measured_points import MeasuredPoint
from iec.passport.passport_model import PassportModel, observed_trial


@dataclass(frozen=True, slots=True, eq=False)
class Design:
    """Design matrix: the conditions of the trials and their outcomes.

    ``run_keys`` marks the point a trial came from: the trials of one point are tied by a
    common random shift, and the fit must sum their contributions by whole points, not one
    by one.

    ``matrix`` is a read-only ``(trials, features)`` array. A tuple of tuples on a journal of
    hundreds of thousands of trials cost seconds on every conversion to an array and tens of
    millions of Python numbers on the way back in the exclusion rule. The constructor takes
    any sequence of rows and converts it itself. Comparison of designs by value is switched
    off (``eq=False``): it is undefined for an array, and no consumer used it.
    """

    matrix: np.ndarray
    outcomes: tuple[int, ...]
    run_keys: tuple[str, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Convert the rows to a two-dimensional array of numbers and forbid editing it."""
        matrix = np.array(self.matrix, dtype=float, copy=True)
        if matrix.ndim != 2:
            matrix = matrix.reshape(len(self.outcomes), len(self.names))
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def n_trials(self) -> int:
        """Return the number of trials in the matrix."""
        return len(self.outcomes)

    @property
    def n_runs(self) -> int:
        """Return the number of points that gave these trials."""
        return len(set(self.run_keys))


def build_profile_design(
    points: Sequence[MeasuredPoint],
    layout: BinLayout,
    profile: PassportModel,
    reference_D: int,  # noqa: N803 - protocol axis
    reference_n: int,
) -> Design:
    """Build the design matrix from the measured points through ``profile``.

    The builder walks the actual Bernoulli trials and calls ``profile.trial_vector(...)`` for
    each. ``Design.names`` is taken only from ``profile.parameter_names(layout)``.

    Args:
        points: Results of the points of the measurement.
        layout: Bin layout of the passport axes.
        profile: The passport model.
        reference_D: Distinct words of the reference point.
        reference_n: Occurrences of each word in the reference point.

    Returns:
        ``Design`` with the trial rows, outcomes and point labels; ``names`` -
        ``profile.parameter_names(layout)``.
    """
    names = profile.parameter_names(layout)
    rows: list[tuple[float, ...]] = []
    outcomes: list[int] = []
    run_keys: list[str] = []
    for point in points:
        if point.status is not PointStatus.OK:
            continue
        for trial in point.trials:
            rows.append(
                profile.trial_vector(
                    # The position shift by one and the exact event number live in the
                    # assembly of coordinates: the record counts the position from zero,
                    # the bins need a number from one.
                    observed_trial(point, trial),
                    layout,
                    reference_D,
                    reference_n,
                )
            )
            outcomes.append(trial.z)
            run_keys.append(point.key)
    return Design(
        matrix=np.array(rows, dtype=float) if rows else np.zeros((0, len(names))),
        outcomes=tuple(outcomes),
        run_keys=tuple(run_keys),
        names=names,
    )


def drop_degenerate_indicators(design: Design) -> tuple[Design, tuple[str, ...]]:
    """Remove the bins without an estimate together with their trials (the exclusion rule).

    A bin is an indicator column of the design. If every trial where it equals one gave a
    right replacement (or every one a wrong one), its coefficient has no likelihood maximum:
    the fit drives it to infinity and drags the neighbours along. Such a column is removed
    together with its trials; removing trials may make another column degenerate, so the
    pass repeats until a fixed point. A column left without trials is removed as well.

    Applies to models whose columns are all indicators (the VS-PMR passport models).

    Returns:
        The pair "reduced design, names of the removed columns in the model order".
    """
    matrix = np.asarray(design.matrix, dtype=float)
    outcomes = np.asarray(design.outcomes, dtype=float)
    if matrix.size == 0:
        return design, ()
    keep_rows = np.ones(len(outcomes), dtype=bool)
    dropped: set[int] = set()
    changed = True
    while changed:
        changed = False
        for column in range(matrix.shape[1]):
            if column in dropped:
                continue
            active = keep_rows & (matrix[:, column] != 0.0)
            if not active.any():
                dropped.add(column)
                changed = True
                continue
            share = float(outcomes[active].mean())
            if share in (0.0, 1.0):
                keep_rows &= ~active
                dropped.add(column)
                changed = True
    keep_columns = [column for column in range(matrix.shape[1]) if column not in dropped]
    reduced = matrix[np.ix_(np.flatnonzero(keep_rows), np.asarray(keep_columns, dtype=np.intp))]
    return (
        Design(
            matrix=reduced,
            outcomes=tuple(int(value) for value in outcomes[keep_rows].tolist()),
            run_keys=tuple(
                key for key, keep in zip(design.run_keys, keep_rows, strict=True) if keep
            ),
            names=tuple(design.names[column] for column in keep_columns),
        ),
        tuple(design.names[column] for column in sorted(dropped)),
    )
