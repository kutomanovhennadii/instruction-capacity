"""The folded rows of a fit: the equal rows of one point folded into one with their trials.

**Folding is exact.** Inside one point all trials share ``D``, ``n`` and ``m``; the design rows
differ only by bins, and many rows are equal. For the logistic model the likelihood, the score
and the information of a row with ``k`` successes out of ``n`` trials equal the sums over its
trials: "ℓ = w·[k·ln μ + (n−k)·ln(1−μ)]", "s = w·(k − n·μ)·x", "I = w·n·μ(1−μ)·x xᵀ". The
shift of a point is shared by all its trials, so the same holds inside the likelihood of the GLMM.

The folded rows are a structure for speed (specification §2.11): they are built once - by
``fold_rows`` from the trials, or by the fit of a quantity from its folded matrix - and every fit
receives them ready. Their key is the trials they were folded from: other trials need new rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FoldedDesign(Protocol):
    """What a fit reads from a design folded inside each point and sorted by points."""

    @property
    def x(self) -> np.ndarray:
        """Return the distinct feature rows ``(rows, features)``."""
        ...

    @property
    def successes(self) -> np.ndarray:
        """Return the successes of every row."""
        ...

    @property
    def trials(self) -> np.ndarray:
        """Return the trials of every row."""
        ...

    @property
    def run(self) -> np.ndarray:
        """Return the point number of every row, non-decreasing."""
        ...

    @property
    def starts(self) -> np.ndarray:
        """Return the indices of the first rows of the points."""
        ...

    @property
    def n_runs(self) -> int:
        """Return the number of points."""
        ...


@dataclass(frozen=True, slots=True)
class FoldedRows:
    """A design folded inside each point and sorted by points.

    ``x`` - distinct feature rows ``(rows, features)``; ``successes`` and ``trials`` - the
    successes and trials of each row; ``run`` - the point number of a row, non-decreasing;
    ``starts`` - indices of the first rows of the points (for ``np.add.reduceat``);
    ``n_runs`` - the number of points. Point numbers are ``0..n_runs−1``, as given by the
    caller.
    """

    x: np.ndarray
    successes: np.ndarray
    trials: np.ndarray
    run: np.ndarray
    starts: np.ndarray
    n_runs: int

    @property
    def n_trials(self) -> int:
        """Return the number of trials before folding."""
        return int(self.trials.sum())


def fold_rows(x: np.ndarray, y: np.ndarray, run: np.ndarray) -> FoldedRows:
    """Fold equal rows of one point into one with a number of trials and successes.

    The row key is the bytes of the pair "point number, feature row": ``np.unique`` over such
    a one-dimensional key is several times cheaper than ``np.unique(axis=0)``. ``-0.0`` and
    ``0.0`` differ as bytes and do not merge - a loss of compression, not an error.

    The keys are built one point at a time: the trials are stably ordered by point number, and
    beyond the input the memory holds a copy of one point. A key built over the whole table at
    once made ``np.unique`` hold three more copies of it: on 3.06 million trials with 112
    features (2.6 GB per copy) the fit failed with ``MemoryError``. The result is the same:
    points in increasing order, the rows of a point in the byte order of their keys.

    Args:
        x: Feature rows ``(trials, features)``.
        y: Outcomes of the trials, 0/1.
        run: Point number of every trial, ``0..n_runs−1``.
    """
    y = np.asarray(y, dtype=float)
    run = np.asarray(run, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        x = x.reshape(len(y), -1)
    x = np.ascontiguousarray(x)
    if len(y) == 0:
        empty = np.zeros(0, dtype=float)
        return FoldedRows(
            x=x,
            successes=empty,
            trials=empty,
            run=np.zeros(0, dtype=np.int64),
            starts=np.zeros(0, dtype=np.int64),
            n_runs=0,
        )
    by_run = np.argsort(run, kind="stable")
    sorted_run = run[by_run]
    bounds = np.flatnonzero(np.r_[True, sorted_run[1:] != sorted_run[:-1], True])
    key = np.dtype((np.void, x.dtype.itemsize * (x.shape[1] + 1)))
    parts_x: list[np.ndarray] = []
    parts_trials: list[np.ndarray] = []
    parts_successes: list[np.ndarray] = []
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        rows = by_run[start:stop]
        keyed = np.ascontiguousarray(np.column_stack([run[rows].astype(float), x[rows]]))
        _, first, inverse = np.unique(
            keyed.view(key).ravel(), return_index=True, return_inverse=True
        )
        inverse = inverse.ravel()
        parts_x.append(x[rows[first]])
        parts_trials.append(np.bincount(inverse, minlength=len(first)).astype(float))
        parts_successes.append(np.bincount(inverse, weights=y[rows], minlength=len(first)))
    folded_run = np.repeat(sorted_run[bounds[:-1]], [len(part) for part in parts_trials])
    starts = np.flatnonzero(np.r_[True, folded_run[1:] != folded_run[:-1]])
    return FoldedRows(
        x=np.concatenate(parts_x),
        successes=np.concatenate(parts_successes),
        trials=np.concatenate(parts_trials),
        run=folded_run,
        starts=starts,
        n_runs=int(len(starts)),
    )
