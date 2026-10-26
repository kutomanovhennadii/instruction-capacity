"""Random-intercept logit on folded rows: the shift of a point inside the likelihood.

The model of trial ``i`` of point ``j``::

    eta_ij = x_ij . theta + b_j,    b_j ~ N(0, sigma2)

The shift ``b_j`` is shared by all trials of the point. It is integrated out of the likelihood
of the point by adaptive Gauss-Hermite quadrature (Liu and Pierce 1994): the nodes of a point
stand around the mode of its shift, spread by the curvature of its log-posterior, and are
placed anew at every evaluation. ``theta`` and ``sigma2`` maximise the weighted marginal
likelihood by Newton steps. The Hessian is the quadrature form of Louis' identity: the
posterior mean of the complete-data Hessian plus the posterior covariance of the complete-data
score. The covariance of the estimates is the cluster sandwich by points around it.

Inside the fit the variance is carried as ``tau = ln sigma``; the result gives ``sigma2`` and
the covariance over ``(theta, sigma2)``. A variance pulled below ``exp(2 * LOG_SIGMA_FLOOR)``
stays there: the points show no spread, the coefficients are those of the plain logit, and
the row and column of ``sigma2`` in the covariance are zero.

**Timing on a measurement of 1 947 points** (37 213 folded rows, 80 coefficients): from the
plain logit 11 steps, 2.2-2.7 s; from the fit of a journal 1 % shorter 7 steps, 1.3 s; a refit
without 20-60 points from the full fit 7-8 steps, 1.5-2.3 s. The sums over the rows of a point
go through the sparse non-zeros of the design (a row has a grid cell and at most one bin of
P, M, R): with dense rows the same fit took 11 s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy import sparse

from iec.passport.logit_fit import inverse_information, logistic_array

#: Quadrature nodes per point. On a measurement of 1 947 points the coefficients with 15 nodes
#: differ from those with 25 by at most 0.022 of their SE; with 11 nodes by 0.09 SE, and with 7
#: nodes Newton does not converge in 100 steps: the quadrature error of the value outgrows the
#: step.
GLMM_NODES = 15
#: Limit of Newton steps of the marginal likelihood.
MAX_STEPS = 100
#: Convergence: the largest change of a parameter in a step (theta in logits, tau = ln sigma).
#: The same threshold as the plain logit: a looser one would let a recomputation stop one step
#: earlier and move the published numbers beyond the tolerance of ``verify``.
STEP_TOLERANCE = 1e-8
#: The floor of ``tau = ln sigma``: the variance does not go below 1e-8.
LOG_SIGMA_FLOOR = math.log(1e-4)
#: Halvings of a Newton step before the line search gives up.
_MAX_HALVINGS = 30
#: Below this Newton decrement ``gradient . step`` (twice the predicted gain of the weighted
#: log-likelihood) the step is taken whole: the quadrature error of the value, about 1e-4 on
#: such a measurement, is then comparable with the gain, and a line search on the value would
#: crawl (36 steps instead of 7 from a warm start).
_FULL_STEP_DECREMENT = 1e-2
#: Steps, tolerance and the largest move of the search of the shift mode of every point.
_MODE_STEPS = 50
_MODE_TOLERANCE = 1e-10
_MODE_STEP_LIMIT = 2.0
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
#: Nodes of the integral over the shift in the threshold of a level.
_THRESHOLD_NODES = 64


class FoldedDesign(Protocol):
    """A design folded inside each point and sorted by points (``logit_fit.FoldedRows``)."""

    @property
    def x(self) -> np.ndarray: ...

    @property
    def successes(self) -> np.ndarray: ...

    @property
    def trials(self) -> np.ndarray: ...

    @property
    def run(self) -> np.ndarray: ...

    @property
    def starts(self) -> np.ndarray: ...

    @property
    def n_runs(self) -> int: ...


@dataclass(frozen=True, slots=True)
class GlmmFit:
    """Result of the fit.

    ``covariance`` - the cluster sandwich over ``(theta, sigma2)``, the last row and column
    belong to ``sigma2``; ``information`` - minus the Hessian over ``(theta, tau)``, the bread
    of the sandwich; ``modes`` - the posterior modes of the point shifts, a warm start of the
    next fit; ``on_floor`` - the variance stayed on its floor and was not estimated.
    """

    theta: np.ndarray
    sigma2: float
    covariance: np.ndarray
    information: np.ndarray
    loglik: float
    modes: np.ndarray
    converged: bool
    on_floor: bool
    steps: int


class _SparseRows:
    """The folded rows with a sparse feature matrix and the sums of rows by point."""

    def __init__(self, rows: FoldedDesign) -> None:
        self.successes = np.asarray(rows.successes, dtype=float)
        self.trials = np.asarray(rows.trials, dtype=float)
        self.run = np.asarray(rows.run)
        self.starts = np.asarray(rows.starts)
        self.n_runs = int(rows.n_runs)
        self.size = int(rows.x.shape[1])
        self.x = sparse.csr_matrix(rows.x)
        coo = self.x.tocoo()
        self._row = coo.row
        self._value = coo.data
        self._slot = self.run[coo.row] * self.size + coo.col

    def run_sums(self, values: np.ndarray) -> np.ndarray:
        """Return ``sum over the rows i of point j of values_i * x_i`` - ``(n_runs, size)``."""
        flat = np.bincount(
            self._slot,
            weights=self._value * values[self._row],
            minlength=self.n_runs * self.size,
        )
        return flat.reshape(self.n_runs, self.size)

    def gram(self, values: np.ndarray) -> np.ndarray:
        """Return ``sum over the rows i of values_i * x_i x_i'`` - ``(size, size)``."""
        return np.asarray((self.x.T @ self.x.multiply(values[:, None])).todense())


@dataclass(frozen=True, slots=True)
class _Evaluation:
    """The weighted marginal log-likelihood and, by order, its gradient and Hessian."""

    value: float
    gradient: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray
    expected_hessian: np.ndarray


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    """Return the Gauss-Hermite nodes and the logarithm of their weights times ``e^(z^2)``.

    The change of variables ``b = mode + sqrt(2) * scale * z`` turns the integral over the
    shift into the Gauss-Hermite form; the factor ``e^(z^2)`` restores the integrand.
    """
    z, w = np.polynomial.hermite.hermgauss(GLMM_NODES)
    return z, np.log(w) + z * z


def _on_floor(tau: float, tau_gradient: float) -> bool:
    """Whether the variance sits on its floor and the likelihood pulls it further down."""
    return tau <= LOG_SIGMA_FLOOR + 1e-12 and tau_gradient <= 0.0


def _modes(
    rows: _SparseRows, eta0: np.ndarray, sigma2: float, start: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mode of the shift posterior of every point and its curvature scale."""
    b = start.copy()
    for _ in range(_MODE_STEPS):
        mu = logistic_array(eta0 + b[rows.run])
        score = np.add.reduceat(rows.successes - rows.trials * mu, rows.starts) - b / sigma2
        info = np.add.reduceat(rows.trials * mu * (1.0 - mu), rows.starts) + 1.0 / sigma2
        step = np.clip(score / info, -_MODE_STEP_LIMIT, _MODE_STEP_LIMIT)
        b += step
        if float(np.max(np.abs(step))) < _MODE_TOLERANCE:
            break
    mu = logistic_array(eta0 + b[rows.run])
    info = np.add.reduceat(rows.trials * mu * (1.0 - mu), rows.starts) + 1.0 / sigma2
    return b, 1.0 / np.sqrt(info)


def _value(
    rows: _SparseRows,
    weight: np.ndarray,
    theta: np.ndarray,
    tau: float,
    mode: np.ndarray,
    scale: np.ndarray,
    quadrature: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the weighted marginal log-likelihood and the pieces its derivatives need.

    The pieces: the shifts at the nodes ``(n_runs, nodes)``, the logits of the rows at the
    nodes ``(rows, nodes)`` and the posterior weights of the nodes ``(n_runs, nodes)``.
    """
    z, log_node = quadrature
    sigma2 = math.exp(2.0 * tau)
    spread = _SQRT2 * scale
    b = mode[:, None] + spread[:, None] * z[None, :]
    eta = (rows.x @ theta)[:, None] + b[rows.run]
    row_loglik = rows.successes[:, None] * eta - rows.trials[:, None] * np.logaddexp(0.0, eta)
    complete = (
        np.add.reduceat(row_loglik, rows.starts, axis=0)
        - b * b / (2.0 * sigma2)
        - tau
        - _HALF_LOG_TWO_PI
    )
    terms = complete + log_node[None, :] + np.log(spread)[:, None]
    top = terms.max(axis=1, keepdims=True)
    mass = np.exp(terms - top)
    total = mass.sum(axis=1, keepdims=True)
    value = float(weight @ (top[:, 0] + np.log(total[:, 0])))
    return value, b, eta, mass / total


def _evaluate(
    rows: _SparseRows,
    weight: np.ndarray,
    theta: np.ndarray,
    tau: float,
    mode: np.ndarray,
    scale: np.ndarray,
    quadrature: tuple[np.ndarray, np.ndarray],
) -> _Evaluation:
    """Return the value, the gradient, the scores of the points and the Hessians."""
    value, b, eta, posterior = _value(rows, weight, theta, tau, mode, scale, quadrature)
    sigma2 = math.exp(2.0 * tau)
    posterior_rows = posterior[rows.run]
    mu = logistic_array(eta)
    residual = rows.successes[:, None] - rows.trials[:, None] * mu
    residual_mean = (residual * posterior_rows).sum(axis=1)
    tau_score = b * b / sigma2 - 1.0
    tau_mean = (posterior * tau_score).sum(axis=1)
    scores = np.column_stack([rows.run_sums(residual_mean), tau_mean])
    gradient = scores.T @ weight
    size = rows.size
    row_weight = weight[rows.run]
    curvature = (posterior_rows * rows.trials[:, None] * mu * (1.0 - mu)).sum(axis=1)
    expected = np.zeros((size + 1, size + 1))
    expected[:size, :size] = -rows.gram(row_weight * curvature)
    expected[size, size] = -2.0 * float(weight @ (posterior * b * b).sum(axis=1)) / sigma2
    spread_term = np.zeros((size + 1, size + 1))
    tau_deviation = tau_score - tau_mean[:, None]
    for node in range(posterior.shape[1]):
        root = np.sqrt(posterior[:, node])
        deviation = (residual[:, node] - residual_mean) * root[rows.run]
        g = np.column_stack([rows.run_sums(deviation), root * tau_deviation[:, node]])
        spread_term += (g * weight[:, None]).T @ g
    return _Evaluation(value, gradient, scores, expected + spread_term, expected)


def fit_glmm(
    rows: FoldedDesign,
    run_weight: np.ndarray,
    theta_start: np.ndarray,
    sigma2_start: float,
    modes_start: np.ndarray | None = None,
) -> GlmmFit:
    """Fit the random-intercept logit by Newton steps on the weighted marginal likelihood.

    Args:
        rows: The folded design (``logit_fit.fold_rows``).
        run_weight: Weight of every point ``(n_runs,)``; zero removes the point from the
            likelihood and from the cluster scores.
        theta_start: Initial coefficients - the plain logit fit or a previous GLMM fit.
        sigma2_start: Initial variance of the shift, positive.
        modes_start: Initial shift modes of the points, or ``None`` - zeros.

    Returns:
        ``GlmmFit``. With ``converged=False`` the estimates and the covariance have no
        meaning - the caller decides what to do.
    """
    design = _SparseRows(rows)
    quadrature = _quadrature()
    weight = np.asarray(run_weight, dtype=float)
    size = design.size
    theta = np.array(theta_start, dtype=float, copy=True)
    tau = max(0.5 * math.log(sigma2_start), LOG_SIGMA_FLOOR)
    start = (
        np.zeros(design.n_runs) if modes_start is None else np.array(modes_start, dtype=float)
    )
    mode, scale = _modes(design, design.x @ theta, math.exp(2.0 * tau), start)
    current = _evaluate(design, weight, theta, tau, mode, scale, quadrature)
    converged = False
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        # On the floor of the variance, with the likelihood pulling it further down, the
        # variance stays there and the step moves the coefficients only.
        free = size if _on_floor(tau, float(current.gradient[size])) else size + 1
        information = -current.hessian[:free, :free]
        try:
            np.linalg.cholesky(information)
        except np.linalg.LinAlgError:
            # Away from the maximum Louis' Hessian may be indefinite; the posterior mean of
            # the complete-data information is positive and still points uphill.
            information = -current.expected_hessian[:free, :free]
        direction = np.zeros(size + 1)
        direction[:free] = inverse_information(information) @ current.gradient[:free]
        if float(np.max(np.abs(direction))) < STEP_TOLERANCE:
            converged = True
            break
        length = 1.0
        whole = float(current.gradient @ direction) < _FULL_STEP_DECREMENT
        accepted = False
        for _ in range(1 if whole else _MAX_HALVINGS):
            candidate_theta = theta + length * direction[:size]
            candidate_tau = max(tau + length * direction[size], LOG_SIGMA_FLOOR)
            candidate_mode, candidate_scale = _modes(
                design, design.x @ candidate_theta, math.exp(2.0 * candidate_tau), mode
            )
            if whole:
                # The quadratic region: the predicted gain is below the quadrature error of
                # the value, so the value cannot judge the step; the step is taken whole.
                accepted = True
                break
            trial, *_ = _value(
                design, weight, candidate_theta, candidate_tau, candidate_mode,
                candidate_scale, quadrature,
            )
            if trial >= current.value - 1e-10 * max(1.0, abs(current.value)):
                accepted = True
                break
            length *= 0.5
        if not accepted:
            break
        theta, tau, mode, scale = candidate_theta, candidate_tau, candidate_mode, candidate_scale
        current = _evaluate(design, weight, theta, tau, mode, scale, quadrature)
        if float(np.max(np.abs(length * direction))) < STEP_TOLERANCE:
            converged = True
            break
    sigma2 = math.exp(2.0 * tau)
    on_floor = _on_floor(tau, float(current.gradient[size]))
    free = size if on_floor else size + 1
    used = weight > 0.0
    groups = int(used.sum())
    observations = int(design.trials[used[design.run]].sum())
    bread = inverse_information(-current.hessian[:free, :free])
    weighted = current.scores[:, :free] * weight[:, None]
    correction = 1.0
    if groups > 1 and observations > free:
        correction = (groups / (groups - 1)) * ((observations - 1) / (observations - free))
    covariance = np.zeros((size + 1, size + 1))
    covariance[:free, :free] = bread @ (weighted.T @ weighted) @ bread * correction
    # From tau = ln sigma to sigma2: d sigma2 / d tau = 2 sigma2.
    jacobian = np.eye(size + 1)
    jacobian[size, size] = 2.0 * sigma2
    covariance = jacobian @ covariance @ jacobian
    return GlmmFit(
        theta=theta,
        sigma2=sigma2,
        covariance=(covariance + covariance.T) / 2.0,
        information=-current.hessian,
        loglik=current.value,
        modes=mode,
        converged=converged,
        on_floor=on_floor,
        steps=steps,
    )


def marginal_threshold(q: float, sigma2: float) -> tuple[float, float]:
    """Return the logit ``eta*`` of the median point at which the share over points is ``q``.

    The share of right replacements over points at the logit ``eta`` of the median point is
    ``p(eta) = E_b[1 / (1 + e^(-(eta + b)))]``, ``b ~ N(0, sigma2)``. The threshold solves
    ``p(eta*) = q``; with ``sigma2 = 0`` it is ``ln(q / (1 - q))``. The second number is
    ``d eta* / d sigma2 = -(d p / d sigma2) / (d p / d eta)`` with the heat identity
    ``d p / d sigma2 = E_b[f''(eta + b)] / 2``, ``f'' = p (1 - p) (1 - 2 p)``.
    """
    eta = math.log(q / (1.0 - q))
    if sigma2 <= 0.0:
        # Without a spread the share is the probability itself: the threshold is the logit of
        # the level exactly, as for a fit without the shift.
        return eta, -0.5 * (1.0 - 2.0 * q)
    z, w = np.polynomial.hermite.hermgauss(_THRESHOLD_NODES)
    weights = w / math.sqrt(math.pi)
    shifts = _SQRT2 * math.sqrt(sigma2) * z
    for _ in range(100):
        p = logistic_array(eta + shifts)
        step = (float(weights @ p) - q) / float(weights @ (p * (1.0 - p)))
        eta -= step
        if abs(step) < 1e-13:
            break
    p = logistic_array(eta + shifts)
    slope = float(weights @ (p * (1.0 - p)))
    curvature = float(weights @ (p * (1.0 - p) * (1.0 - 2.0 * p)))
    return eta, -0.5 * curvature / slope
