"""Logistic fit by Newton steps on folded rows, with the cluster sandwich covariance.

**Folding is exact.** Inside one point all trials share ``D``, ``n`` and ``m``; the design rows
differ only by bins, and many rows are equal. For the logistic model the likelihood, the score
and the information of a row with ``k`` successes out of ``n`` trials equal the sums over its
trials: "ℓ = w·[k·ln μ + (n−k)·ln(1−μ)]", "s = w·(k − n·μ)·x", "I = w·n·μ(1−μ)·x xᵀ". The
finite-sample correction of the sandwich takes the number of trials, not of rows.

**Newton.** The step "Δθ = I⁺·s" uses the pseudo-inverse with the threshold
``INFORMATION_RCOND`` - the same as the bread of the sandwich. Convergence: the largest change
of a coefficient is below ``STEP_TOLERANCE``. Separation of the outcome - every row unanimous
and μ̂ equal to the outcome - is flagged and counts as converged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logistic_array(scores: np.ndarray) -> np.ndarray:
    """Turn a vector of scores into probabilities: 1/(1 + e^(−score)), stable for large scores.

    Stability is in the branch by sign: for a large positive score ``exp(-score)`` goes to
    zero without overflow, for a large negative one ``exp(-score)`` would overflow, so
    ``exp(score)`` is computed there. Every element equals the scalar formula bit for bit.
    """
    scores = np.asarray(scores, dtype=float)
    result = np.empty_like(scores)
    positive = scores >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    weight = np.exp(scores[~positive])
    result[~positive] = weight / (1.0 + weight)
    return result


#: Threshold of data support of a direction when the information matrix is inverted. A
#: direction with an eigenvalue below ``INFORMATION_RCOND`` of the largest one is not supported
#: by the data: inverting it divides noise by noise. The ``pinv`` default (about 1e-15) is too
#: generous for this: at a condition number near 1e15 it inverts every direction and gives
#: **negative variances** - a matrix that is no longer a covariance. With this threshold only
#: numerically empty directions are cut, and the result is a correct covariance of minimum
#: norm. The real cure for missing directions is not a threshold but data: there must be more
#: informative points than coefficients.
INFORMATION_RCOND = 1e-12


def inverse_information(information: np.ndarray, *, rcond: float = INFORMATION_RCOND) -> np.ndarray:
    """Invert only the data-supported directions of a symmetric information matrix."""
    return np.linalg.pinv(information, rcond=rcond, hermitian=True)


def binomial_cluster_covariance(
    *,
    exog: np.ndarray,
    endog: np.ndarray,
    groups: np.ndarray,
    params: np.ndarray,
    mean: np.ndarray,
    derivative: np.ndarray,
    variance: np.ndarray,
    observations: int | None = None,
) -> np.ndarray:
    """Assemble the sandwich covariance for an arbitrary binomial link.

    ``derivative`` is ``d mu / d eta``. Then the expected information has the weights
    ``derivative**2 / variance``, and the score is ``x * (y-mu) * derivative/variance``.

    Args:
        observations: Number of observations for the finite-cluster correction; by default
            the rows of ``exog``. Given when the rows are **folded**: in the binomial form one
            row carries several trials, and a correction computed over folded rows would
            differ from the correction of the same problem in the Bernoulli form.
    """
    del params  # kept in the contract: the calling boundary fixes the state of the fit
    safe_variance = np.maximum(variance, np.finfo(float).tiny)
    weights = np.divide(
        derivative**2,
        safe_variance,
        out=np.zeros_like(derivative),
        where=safe_variance > 0.0,
    )
    information = exog.T @ (exog * weights[:, None])
    bread = inverse_information(information)

    score_factor = np.divide(
        (endog - mean) * derivative,
        safe_variance,
        out=np.zeros_like(mean),
        where=safe_variance > 0.0,
    )
    # The scores of the points are sums over segments of sorted rows (``np.add.reduceat``),
    # and the "meat" is one product ``SᵀS``. A loop with a boolean mask over all rows for
    # every point cost seconds on thousands of points and hundreds of thousands of rows.
    groups = np.asarray(groups)
    if len(groups) == 0:
        meat = np.zeros((exog.shape[1], exog.shape[1]), dtype=float)
        n_groups = 0
    else:
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        starts = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
        scores = np.add.reduceat((exog * score_factor[:, None])[order], starts, axis=0)
        meat = scores.T @ scores
        n_groups = len(starts)
    n_rows, n_params = exog.shape
    if observations is not None:
        n_rows = observations
    correction = 1.0
    if n_groups > 1:
        correction = (n_groups / (n_groups - 1)) * ((n_rows - 1) / (n_rows - n_params))
    covariance = bread @ meat @ bread * correction
    return (covariance + covariance.T) / 2.0


#: Limit of Newton steps: about ten steps from zero on eighty coefficients, with a fivefold
#: margin.
MAX_STEPS = 55
#: Convergence - the largest change of a coefficient in a step, in logits.
STEP_TOLERANCE = 1e-8


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
    a one-dimensional key is several times cheaper than ``np.unique(axis=0)``. The byte order
    is not the order of the numbers, so the folded rows are then sorted by point explicitly.
    ``-0.0`` and ``0.0`` differ as bytes and do not merge - a loss of compression, not an
    error.

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
    keyed = np.ascontiguousarray(np.column_stack([run.astype(float), x]))
    view = keyed.view(np.dtype((np.void, keyed.dtype.itemsize * keyed.shape[1]))).ravel()
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    folded_run = run[first]
    order = np.argsort(folded_run, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    inverse = rank[inverse]
    trials = np.bincount(inverse, minlength=len(first)).astype(float)
    successes = np.bincount(inverse, weights=y, minlength=len(first))
    folded_run = folded_run[order]
    starts = np.flatnonzero(np.r_[True, folded_run[1:] != folded_run[:-1]])
    return FoldedRows(
        x=np.ascontiguousarray(x[first][order]),
        successes=successes,
        trials=trials,
        run=folded_run,
        starts=starts,
        n_runs=int(len(starts)),
    )


@dataclass(frozen=True, slots=True)
class LogitFit:
    """Result of the fit: estimates, the information ``X'WX`` at them, the sandwich, flags, steps.

    ``information`` is the inverse of the naive covariance.
    """

    theta: np.ndarray
    information: np.ndarray
    covariance: np.ndarray
    converged: bool
    separated: bool
    steps: int


def fit_logit(
    rows: FoldedRows,
    run_weight: np.ndarray,
    start: np.ndarray | None = None,
    *,
    max_steps: int = MAX_STEPS,
    tolerance: float = STEP_TOLERANCE,
) -> LogitFit:
    """Fit the weighted logit by Newton and assemble the cluster sandwich by points.

    Args:
        rows: The folded design.
        run_weight: Weight of every point ``(n_runs,)``. Zero - the point is removed: its rows
            enter neither the likelihood nor the cluster scores.
        start: Initial estimates in column order, or ``None`` - zeros.
        max_steps: Step limit; exhausting it gives ``converged=False``.
        tolerance: Convergence threshold on the largest change of a coefficient.

    Returns:
        ``LogitFit``. With ``converged=False`` or non-numeric estimates the covariance has no
        meaning - the caller decides what to do.
    """
    run_weight = np.asarray(run_weight, dtype=float)
    kept = run_weight[rows.run] > 0.0
    x = rows.x[kept]
    n = rows.trials[kept]
    y = rows.successes[kept] / n
    w = run_weight[rows.run][kept] * n
    group = rows.run[kept]
    theta = (
        np.zeros(rows.x.shape[1]) if start is None else np.array(start, dtype=float, copy=True)
    )
    unanimous = bool(np.all((y == 0.0) | (y == 1.0)))
    converged = separated = False
    steps = 0
    for steps in range(1, max_steps + 1):
        mu = logistic_array(x @ theta)
        information = x.T @ (x * (w * mu * (1.0 - mu))[:, None])
        step = inverse_information(information) @ (x.T @ (w * (y - mu)))
        theta = theta + step
        if not np.all(np.isfinite(theta)):
            break
        if unanimous and np.allclose(logistic_array(x @ theta) - y, 0.0):
            # Separation: the outcome is predicted exactly; the fit stops as converged.
            separated = converged = True
            break
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
    mu = logistic_array(x @ theta)
    variance = w * mu * (1.0 - mu)
    information = x.T @ (x * variance[:, None])
    covariance = binomial_cluster_covariance(
        exog=x,
        endog=w * y,
        groups=group,
        params=theta,
        mean=w * mu,
        derivative=variance,
        variance=variance,
        observations=int(n.sum()),
    )
    return LogitFit(
        theta=theta,
        information=information,
        covariance=covariance,
        converged=converged,
        separated=separated,
        steps=steps,
    )
