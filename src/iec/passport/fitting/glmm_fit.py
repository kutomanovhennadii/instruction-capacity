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

The spread may depend on the load: ``b_j ~ N(0, sigma_j^2)``, ``ln sigma_j = z_j . gamma``
(``GlmmFitter.fit_spread``, the rows ``z_j`` come from :mod:`iec.passport.fitting.load_spread`).
The Newton steps are the same; one variance is the case of one column of ones, with
``gamma = tau``.

**Setup and work.** ``GlmmFitter`` receives the sparse design of the folded rows and the
quadrature rule ready (``acceleration``): both are built once, and the fitter fits from as many
starts and with as many spreads as the caller needs without building them again. ``fit_glmm``
and ``fit_spread`` make one fit and build the sparse design of the rows for that call.

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

from iec.passport.fitting.acceleration.folded_rows import FoldedDesign
from iec.passport.fitting.acceleration.quadrature_rule import QuadratureRule, build_quadrature_rule
from iec.passport.fitting.acceleration.sparse_design import SparseDesign, build_sparse_design
from iec.passport.fitting.logit_fit import inverse_information, logistic_array

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
#: crawl (36 steps instead of 7 from a warm start). The quadrature error grows with the value, so
#: on a larger measurement the threshold is a share of the value.
_FULL_STEP_DECREMENT = 1e-2
_FULL_STEP_DECREMENT_SHARE = 1e-6
#: Steps, tolerance and the largest move of the search of the shift mode of every point.
_MODE_STEPS = 50
_MODE_TOLERANCE = 1e-10
_MODE_STEP_LIMIT = 2.0
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
#: Nodes of the integral over the shift in the threshold of a level.
_THRESHOLD_NODES = 64
#: The quadrature rule of the marginal likelihood, built once when the module is loaded; its key
#: is ``GLMM_NODES``.
GLMM_RULE = build_quadrature_rule(GLMM_NODES)
#: The quadrature rule of the threshold of a level, built once; its key is ``_THRESHOLD_NODES``.
_THRESHOLD_RULE = build_quadrature_rule(_THRESHOLD_NODES)


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


@dataclass(frozen=True, slots=True)
class SpreadFit:
    """Result of the fit with the spread by the load.

    ``covariance`` - the cluster sandwich over ``(theta, gamma)``; ``information`` - minus the
    Hessian over them; ``modes`` - the posterior modes of the point shifts; ``floored`` - the fit
    stopped unconverged because every halving of its last step took the spread of some point below
    the floor (``load_spread.fit_spread_merging`` then merges the group of the lowest spread).
    """

    theta: np.ndarray
    gamma: np.ndarray
    covariance: np.ndarray
    information: np.ndarray
    loglik: float
    modes: np.ndarray
    converged: bool
    steps: int
    floored: bool = False


@dataclass(frozen=True, slots=True)
class _Evaluation:
    """The weighted marginal log-likelihood and, by order, its gradient and Hessian."""

    value: float
    gradient: np.ndarray
    scores: np.ndarray
    hessian: np.ndarray


def _on_floor(tau: float, tau_gradient: float) -> bool:
    """Whether the variance sits on its floor and the likelihood pulls it further down."""
    return tau <= LOG_SIGMA_FLOOR + 1e-12 and tau_gradient <= 0.0


class _VarianceFloor(Protocol):
    """How the floor of the variance acts on a Newton step."""

    def pinned(self, gamma: np.ndarray, gradient: np.ndarray, size: int) -> bool:
        """Return whether the variance stays on its floor: the step moves the coefficients only."""
        ...

    def admit(self, gamma: np.ndarray, spread_x: np.ndarray) -> np.ndarray | None:
        """Return the parameters of the spread a step may take, or ``None``: the step is halved."""
        ...


class _OneVarianceFloor:
    """One variance: on its floor it stays there, and a step below the floor is cut at it."""

    def pinned(self, gamma: np.ndarray, gradient: np.ndarray, size: int) -> bool:
        """Return whether ``tau`` sits on its floor and the likelihood pulls it further down."""
        return _on_floor(float(gamma[0]), float(gradient[size]))

    def admit(self, gamma: np.ndarray, spread_x: np.ndarray) -> np.ndarray | None:
        """Return ``tau`` cut at its floor."""
        return np.maximum(gamma, LOG_SIGMA_FLOOR)


class _SpreadFloor:
    """The spread by the load: a step taking the spread of a point below the floor is halved."""

    def pinned(self, gamma: np.ndarray, gradient: np.ndarray, size: int) -> bool:
        """Return ``False``: the parameters of the spread are always estimated."""
        return False

    def admit(self, gamma: np.ndarray, spread_x: np.ndarray) -> np.ndarray | None:
        """Return ``gamma`` when every point keeps its spread above the floor, else ``None``."""
        if float((spread_x @ gamma).min()) < LOG_SIGMA_FLOOR:
            return None
        return gamma


_ONE_VARIANCE_FLOOR: _VarianceFloor = _OneVarianceFloor()
_SPREAD_FLOOR: _VarianceFloor = _SpreadFloor()


def _floor_of(spread_x: np.ndarray) -> _VarianceFloor:
    """Return the rule of the floor: one column of the spread is one variance."""
    return _ONE_VARIANCE_FLOOR if spread_x.shape[1] == 1 else _SPREAD_FLOOR


def _modes(
    rows: SparseDesign, eta0: np.ndarray, sigma2: float | np.ndarray, start: np.ndarray
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
    rows: SparseDesign,
    weight: np.ndarray,
    theta: np.ndarray,
    log_sigma: np.ndarray,
    mode: np.ndarray,
    scale: np.ndarray,
    quadrature: QuadratureRule,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the weighted marginal log-likelihood and the pieces its derivatives need.

    ``log_sigma`` - ``ln sigma`` of every point. The pieces: the shifts at the nodes
    ``(n_runs, nodes)``, the logits of the rows at the nodes ``(rows, nodes)`` and the posterior
    weights of the nodes ``(n_runs, nodes)``.
    """
    z, log_node = quadrature.nodes, quadrature.log_weights
    sigma2 = np.exp(2.0 * log_sigma)
    spread = _SQRT2 * scale
    b = mode[:, None] + spread[:, None] * z[None, :]
    eta = (rows.x @ theta)[:, None] + b[rows.run]
    row_loglik = rows.successes[:, None] * eta - rows.trials[:, None] * np.logaddexp(0.0, eta)
    complete = (
        np.add.reduceat(row_loglik, rows.starts, axis=0)
        - b * b / (2.0 * sigma2[:, None])
        - log_sigma[:, None]
        - _HALF_LOG_TWO_PI
    )
    terms = complete + log_node[None, :] + np.log(spread)[:, None]
    top = terms.max(axis=1, keepdims=True)
    mass = np.exp(terms - top)
    total = mass.sum(axis=1, keepdims=True)
    value = float(weight @ (top[:, 0] + np.log(total[:, 0])))
    return value, b, eta, mass / total


def _node_terms(
    rows: SparseDesign,
    theta: np.ndarray,
    log_sigma: np.ndarray,
    spread_x: np.ndarray,
    mode: np.ndarray,
    scale: np.ndarray,
    quadrature: QuadratureRule,
    b: np.ndarray,
    residual: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    """Return what the move of the quadrature nodes adds to the scores of the points.

    The nodes of point ``j`` stand at ``b_jk = m_j + sqrt(2) s_j z_k``: ``m_j`` is the mode of its
    shift, ``s_j = 1 / sqrt(I_j)``, ``I_j`` the curvature of its complete log-likelihood in the shift
    at the mode. Both are placed anew at every evaluation, so the derivative of the value that the
    line search compares is::

        d l_j / d psi = (nodes held) + d l_j / d m_j * d m_j / d psi + d l_j / d s_j * d s_j / d psi

    with ``d l_j / d m_j = sum_k p_jk c'_jk`` and ``d l_j / d s_j = sum_k p_jk c'_jk sqrt(2) z_k +
    1 / s_j``, where ``c'_jk`` is the derivative of the complete log-likelihood in the shift at node
    ``k`` and ``p_jk`` the posterior weight of the node; ``d m_j / d psi = -(d2 c_j / d b2)^-1 *
    d2 c_j / d b d psi`` and ``d s_j / d psi = -I_j^(-3/2) / 2 * d I_j / d psi``. The first term is
    the score of ``_evaluate`` with the nodes held; without the other two a step went by a
    derivative of another function than the value it was judged by, and the fits crawled for
    113-400 steps (decision 3 of the passport standard). ``residual`` are the residuals of the rows at
    the nodes that ``_evaluate`` has already computed. Returns ``(n_runs, size + len(gamma))``.
    """
    z = quadrature.nodes
    sigma2 = np.exp(2.0 * log_sigma)
    slope = np.add.reduceat(residual, rows.starts, axis=0) - b / sigma2[:, None]
    along_mode = (posterior * slope).sum(axis=1)
    along_scale = (posterior * slope * (_SQRT2 * z)[None, :]).sum(axis=1) + 1.0 / scale
    centre = logistic_array(rows.x @ theta + mode[rows.run])
    variance = rows.trials * centre * (1.0 - centre)
    skew = variance * (1.0 - 2.0 * centre)
    information = np.add.reduceat(variance, rows.starts) + 1.0 / sigma2
    skew_sum = np.add.reduceat(skew, rows.starts)
    mode_theta = -rows.run_sums(variance) / information[:, None]
    mode_gamma = (2.0 * mode / sigma2 / information)[:, None] * spread_x
    info_theta = rows.run_sums(skew) + skew_sum[:, None] * mode_theta
    info_gamma = (-2.0 / sigma2)[:, None] * spread_x + skew_sum[:, None] * mode_gamma
    factor = -0.5 * information**-1.5
    return np.column_stack(
        [
            along_mode[:, None] * mode_theta + (along_scale * factor)[:, None] * info_theta,
            along_mode[:, None] * mode_gamma + (along_scale * factor)[:, None] * info_gamma,
        ]
    )


def _evaluate(
    rows: SparseDesign,
    weight: np.ndarray,
    theta: np.ndarray,
    log_sigma: np.ndarray,
    spread_x: np.ndarray,
    mode: np.ndarray,
    scale: np.ndarray,
    quadrature: QuadratureRule,
) -> _Evaluation:
    """Return the value, the gradient, the scores of the points and the Hessian.

    The scores and the gradient are the total derivatives of the value, with the move of the
    quadrature nodes (``_node_terms``). The Hessian is over ``(theta, gamma)``: the posterior mean of
    the complete-data Hessian plus the posterior covariance of the complete-data score with the
    nodes held. The posterior covariance of the complete-data score
    goes node by node only in its block over ``theta``. The score over ``gamma`` of node ``k`` of
    point ``j`` is ``D_jk z_j``, the deviation of the score of ``ln sigma`` times the row of the
    point, so its blocks are summed over the nodes inside the point: ``z' diag(w sum_k p D^2) z``
    and ``run_sums(sum_k p D (r - r_mean))' diag(w) z``. The Hessian is the same to 1e-16 of its
    largest element; with 25 columns of ``z`` on a synthetic table of 3 500 points a step is 7 %
    faster than with ``z`` inside the loop over the nodes.
    """
    value, b, eta, posterior = _value(rows, weight, theta, log_sigma, mode, scale, quadrature)
    sigma2 = np.exp(2.0 * log_sigma)
    posterior_rows = posterior[rows.run]
    mu = logistic_array(eta)
    residual = rows.successes[:, None] - rows.trials[:, None] * mu
    residual_mean = (residual * posterior_rows).sum(axis=1)
    tau_score = b * b / sigma2[:, None] - 1.0
    tau_mean = (posterior * tau_score).sum(axis=1)
    scores = np.column_stack([rows.run_sums(residual_mean), tau_mean[:, None] * spread_x])
    scores = scores + _node_terms(
        rows, theta, log_sigma, spread_x, mode, scale, quadrature, b, residual, posterior
    )
    gradient = scores.T @ weight
    size = rows.size
    total = size + spread_x.shape[1]
    row_weight = weight[rows.run]
    curvature = (posterior_rows * rows.trials[:, None] * mu * (1.0 - mu)).sum(axis=1)
    expected = np.zeros((total, total))
    expected[:size, :size] = -rows.gram(row_weight * curvature)
    second = 2.0 * (posterior * b * b).sum(axis=1) / sigma2
    expected[size:, size:] = -(spread_x * (weight * second)[:, None]).T @ spread_x
    spread_term = np.zeros((total, total))
    deviation = residual - residual_mean[:, None]
    tau_deviation = tau_score - tau_mean[:, None]
    for node in range(posterior.shape[1]):
        u = rows.run_sums(deviation[:, node] * np.sqrt(posterior[:, node])[rows.run])
        spread_term[:size, :size] += (u * weight[:, None]).T @ u
    cross = rows.run_sums((deviation * (posterior * tau_deviation)[rows.run]).sum(axis=1))
    spread_term[:size, size:] = (cross * weight[:, None]).T @ spread_x
    spread_term[size:, :size] = spread_term[:size, size:].T
    gamma_weight = weight * (posterior * tau_deviation * tau_deviation).sum(axis=1)
    spread_term[size:, size:] = (spread_x * gamma_weight[:, None]).T @ spread_x
    return _Evaluation(value, gradient, scores, expected + spread_term)


@dataclass(frozen=True, slots=True)
class _Maximum:
    """Where the Newton steps stopped; ``floored`` - stopped by the floor of the spread."""

    theta: np.ndarray
    gamma: np.ndarray
    mode: np.ndarray
    current: _Evaluation
    converged: bool
    steps: int
    floored: bool


@dataclass(frozen=True, slots=True)
class _Refusal:
    """Every halving of a step was refused; ``floored`` - each of them by the floor of the spread."""

    floored: bool


@dataclass(frozen=True, slots=True)
class _Move:
    """An accepted Newton step: the new parameters, the shift modes, their scales, the length.

    ``cut`` - the floor of the spread refused a longer candidate of this step.
    """

    theta: np.ndarray
    gamma: np.ndarray
    mode: np.ndarray
    scale: np.ndarray
    length: float
    cut: bool = False


def _newton_direction(
    current: _Evaluation, floor: _VarianceFloor, gamma: np.ndarray, size: int, count: int
) -> np.ndarray:
    """Return the Newton direction over ``(theta, gamma)`` at the current evaluation.

    On the floor of one variance the direction moves the coefficients only. Away from the
    maximum Louis' Hessian may be indefinite; then its eigenvalues are taken by their moduli.
    """
    pinned = floor.pinned(gamma, current.gradient, size)
    free = size if pinned else size + count
    information = -current.hessian[:free, :free]
    try:
        np.linalg.cholesky(information)
    except np.linalg.LinAlgError:
        # Louis' Hessian may be indefinite away from the maximum; its eigenvalues are taken by
        # their moduli (not below 1e-6 of the largest), the eigenvectors stay: along a positive
        # curvature the step stays Newton's, along a negative one it goes uphill with a bounded
        # length (decision 3 of the passport standard; the expected information crawled).
        values, vectors = np.linalg.eigh((information + information.T) / 2.0)
        values = np.maximum(np.abs(values), 1e-6 * float(np.max(np.abs(values))))
        information = (vectors * values) @ vectors.T
    direction = np.zeros(size + count)
    direction[:free] = inverse_information(information) @ current.gradient[:free]
    return direction


def _line_search(
    design: SparseDesign,
    quadrature: QuadratureRule,
    floor: _VarianceFloor,
    weight: np.ndarray,
    spread_x: np.ndarray,
    theta: np.ndarray,
    gamma: np.ndarray,
    mode: np.ndarray,
    current: _Evaluation,
    direction: np.ndarray,
) -> _Move | _Refusal:
    """Return the step accepted along ``direction``, or the refusal of every halving.

    Below the Newton decrement ``_FULL_STEP_DECREMENT`` the step is taken whole: the quadrature
    error of the value is then comparable with the predicted gain. Otherwise the step is halved
    until the value does not fall, at most ``_MAX_HALVINGS`` times; the floor of the variance
    admits or halves the candidate first. The refusal tells whether the floor refused every
    candidate or the value fell; an accepted step tells whether the floor cut it.
    """
    size = design.size
    length = 1.0
    admitted = False
    refused = False
    whole = float(current.gradient @ direction) < max(
        _FULL_STEP_DECREMENT, _FULL_STEP_DECREMENT_SHARE * abs(current.value)
    )
    for _ in range(1 if whole else _MAX_HALVINGS):
        candidate_theta = theta + length * direction[:size]
        candidate_gamma = floor.admit(gamma + length * direction[size:], spread_x)
        if candidate_gamma is None:
            refused = True
            length *= 0.5
            continue
        admitted = True
        candidate_mode, candidate_scale = _modes(
            design,
            design.x @ candidate_theta,
            np.exp(2.0 * (spread_x @ candidate_gamma)),
            mode,
        )
        move = _Move(
            candidate_theta, candidate_gamma, candidate_mode, candidate_scale, length, refused
        )
        if whole:
            # The quadratic region: the predicted gain is below the quadrature error of
            # the value, so the value cannot judge the step; the step is taken whole.
            return move
        trial, *_ = _value(
            design, weight, candidate_theta, spread_x @ candidate_gamma, candidate_mode,
            candidate_scale, quadrature,
        )
        if trial >= current.value - 1e-10 * max(1.0, abs(current.value)):
            return move
        length *= 0.5
    return _Refusal(floored=not admitted)


def _maximise(
    design: SparseDesign,
    quadrature: QuadratureRule,
    weight: np.ndarray,
    theta_start: np.ndarray,
    gamma_start: np.ndarray,
    spread_x: np.ndarray,
    modes_start: np.ndarray | None,
) -> _Maximum:
    """Run the Newton steps over ``(theta, gamma)``.

    The floor of the variance holds for one variance, the case of one column: there the variance
    stays on it and the step moves the coefficients only. With several columns a step that takes
    the spread of some point below the floor is halved (``_floor_of``).
    """
    floor = _floor_of(spread_x)
    size = design.size
    count = spread_x.shape[1]
    theta = np.array(theta_start, dtype=float, copy=True)
    gamma = np.array(gamma_start, dtype=float, copy=True)
    start = (
        np.zeros(design.n_runs) if modes_start is None else np.array(modes_start, dtype=float)
    )
    mode, scale = _modes(design, design.x @ theta, np.exp(2.0 * (spread_x @ gamma)), start)
    current = _evaluate(design, weight, theta, spread_x @ gamma, spread_x, mode, scale, quadrature)
    converged = False
    floored = False
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        direction = _newton_direction(current, floor, gamma, size, count)
        if float(np.max(np.abs(direction))) < STEP_TOLERANCE:
            converged = True
            break
        move = _line_search(
            design, quadrature, floor, weight, spread_x, theta, gamma, mode, current, direction
        )
        if isinstance(move, _Refusal):
            floored = move.floored
            break
        theta, gamma, mode, scale = move.theta, move.gamma, move.mode, move.scale
        current = _evaluate(
            design, weight, theta, spread_x @ gamma, spread_x, mode, scale, quadrature
        )
        if float(np.max(np.abs(move.length * direction))) < STEP_TOLERANCE:
            # A step that the floor of the spread cut to nothing is no convergence: the maximum
            # lies beyond the floor, and the fit stopped on it (decision 4 of the standard).
            converged = not move.cut
            floored = move.cut
            break
    return _Maximum(theta, gamma, mode, current, converged, steps, floored)


def _sandwich(
    design: SparseDesign, weight: np.ndarray, current: _Evaluation, free: int
) -> np.ndarray:
    """Return the cluster sandwich by points over the first ``free`` parameters."""
    used = weight > 0.0
    groups = int(used.sum())
    observations = int(design.trials[used[design.run]].sum())
    bread = inverse_information(-current.hessian[:free, :free])
    weighted = current.scores[:, :free] * weight[:, None]
    correction = 1.0
    if groups > 1 and observations > free:
        correction = (groups / (groups - 1)) * ((observations - 1) / (observations - free))
    result: np.ndarray = bread @ (weighted.T @ weighted) @ bread * correction
    return result


class GlmmFitter:
    """The fits of the GLMM on one folded design.

    The sparse design of the rows and the quadrature rule are structures for speed (specification
    §2.11): they are built once and handed to the fitter ready, and the fitter fits as many times
    as the caller needs - from several starts of the variance and with the spread by the load -
    without building them again.
    """

    def __init__(self, design: SparseDesign, quadrature: QuadratureRule = GLMM_RULE) -> None:
        """Keep the structures of the fits.

        Args:
            design: The sparse design of the folded rows (``build_sparse_design``).
            quadrature: The quadrature rule of the marginal likelihood; by default ``GLMM_RULE``.
        """
        self._design = design
        self._quadrature = quadrature

    def fit(
        self,
        run_weight: np.ndarray,
        theta_start: np.ndarray,
        sigma2_start: float,
        modes_start: np.ndarray | None = None,
    ) -> GlmmFit:
        """Fit the random-intercept logit by Newton steps on the weighted marginal likelihood.

        Args:
            run_weight: Weight of every point ``(n_runs,)``; zero removes the point from the
                likelihood and from the cluster scores.
            theta_start: Initial coefficients - the plain logit fit or a previous GLMM fit.
            sigma2_start: Initial variance of the shift, positive.
            modes_start: Initial shift modes of the points, or ``None`` - zeros.

        Returns:
            ``GlmmFit``. With ``converged=False`` the estimates and the covariance have no
            meaning - the caller decides what to do.
        """
        design = self._design
        weight = np.asarray(run_weight, dtype=float)
        size = design.size
        tau_start = max(0.5 * math.log(sigma2_start), LOG_SIGMA_FLOOR)
        found = _maximise(
            design,
            self._quadrature,
            weight,
            theta_start,
            np.array([tau_start]),
            np.ones((design.n_runs, 1)),
            modes_start,
        )
        tau = float(found.gamma[0])
        sigma2 = math.exp(2.0 * tau)
        on_floor = _on_floor(tau, float(found.current.gradient[size]))
        free = size if on_floor else size + 1
        covariance = np.zeros((size + 1, size + 1))
        covariance[:free, :free] = _sandwich(design, weight, found.current, free)
        # From tau = ln sigma to sigma2: d sigma2 / d tau = 2 sigma2.
        jacobian = np.eye(size + 1)
        jacobian[size, size] = 2.0 * sigma2
        covariance = jacobian @ covariance @ jacobian
        return GlmmFit(
            theta=found.theta,
            sigma2=sigma2,
            covariance=(covariance + covariance.T) / 2.0,
            information=-found.current.hessian,
            loglik=found.current.value,
            modes=found.mode,
            converged=found.converged,
            on_floor=on_floor,
            steps=found.steps,
        )

    def fit_spread(
        self,
        run_weight: np.ndarray,
        theta_start: np.ndarray,
        gamma_start: np.ndarray,
        spread_x: np.ndarray,
        modes_start: np.ndarray | None = None,
    ) -> SpreadFit:
        """Fit the logit with the spread by the load: ``ln sigma_j = z_j . gamma``.

        Args:
            run_weight: Weight of every point; zero removes the point.
            theta_start: Initial coefficients, usually those of the fit with one variance.
            gamma_start: Initial parameters of the spread.
            spread_x: The rows ``z_j`` of the points ``(n_runs, len(gamma))``.
            modes_start: Initial shift modes of the points, or ``None`` - zeros.

        Returns:
            ``SpreadFit``. With ``converged=False`` the estimates have no meaning.
        """
        design = self._design
        weight = np.asarray(run_weight, dtype=float)
        matrix = np.asarray(spread_x, dtype=float)
        found = _maximise(
            design, self._quadrature, weight, theta_start, gamma_start, matrix, modes_start
        )
        covariance = _sandwich(design, weight, found.current, design.size + matrix.shape[1])
        return SpreadFit(
            theta=found.theta,
            gamma=found.gamma,
            covariance=(covariance + covariance.T) / 2.0,
            information=-found.current.hessian,
            loglik=found.current.value,
            modes=found.mode,
            converged=found.converged,
            steps=found.steps,
            floored=found.floored,
        )


def fit_glmm(
    rows: FoldedDesign,
    run_weight: np.ndarray,
    theta_start: np.ndarray,
    sigma2_start: float,
    modes_start: np.ndarray | None = None,
) -> GlmmFit:
    """Make one fit of the random-intercept logit on folded rows (``GlmmFitter.fit``).

    The sparse design of the rows is built for this call; repeated fits on the same rows
    assemble one ``GlmmFitter`` and reuse it.

    Args:
        rows: The folded design (``acceleration.folded_rows``).
        run_weight: Weight of every point ``(n_runs,)``; zero removes the point from the
            likelihood and from the cluster scores.
        theta_start: Initial coefficients - the plain logit fit or a previous GLMM fit.
        sigma2_start: Initial variance of the shift, positive.
        modes_start: Initial shift modes of the points, or ``None`` - zeros.

    Returns:
        ``GlmmFit``. With ``converged=False`` the estimates and the covariance have no
        meaning - the caller decides what to do.
    """
    fitter = GlmmFitter(build_sparse_design(rows))
    return fitter.fit(run_weight, theta_start, sigma2_start, modes_start)


def fit_spread(
    rows: FoldedDesign,
    run_weight: np.ndarray,
    theta_start: np.ndarray,
    gamma_start: np.ndarray,
    spread_x: np.ndarray,
    modes_start: np.ndarray | None = None,
) -> SpreadFit:
    """Make one fit of the logit with the spread by the load (``GlmmFitter.fit_spread``).

    The sparse design of the rows is built for this call.

    Args:
        rows: The folded design.
        run_weight: Weight of every point; zero removes the point.
        theta_start: Initial coefficients, usually those of the fit with one variance.
        gamma_start: Initial parameters of the spread.
        spread_x: The rows ``z_j`` of the points ``(n_runs, len(gamma))``.
        modes_start: Initial shift modes of the points, or ``None`` - zeros.

    Returns:
        ``SpreadFit``. With ``converged=False`` the estimates have no meaning.
    """
    fitter = GlmmFitter(build_sparse_design(rows))
    return fitter.fit_spread(run_weight, theta_start, gamma_start, spread_x, modes_start)


def threshold_estimate(q: float, sigma2: float) -> float:
    """Return the estimate ``ln(q / (1 - q)) * sqrt(1 + 3 sigma2 / pi^2)`` of the threshold.

    The threshold of :func:`marginal_threshold` has no closed form. This formula is exact for a
    bridge-distributed shift (Wang and Louis 2003); for the normal shift it is within 1.4 % at
    ``q = 0.95`` for ``sigma2`` from 0.5 to 50. It serves for an estimate by hand and as the start
    of Newton's method: the numbers of a passport come from the root alone.
    """
    return math.log(q / (1.0 - q)) * math.sqrt(1.0 + 3.0 * max(sigma2, 0.0) / math.pi**2)


def marginal_threshold(q: float, sigma2: float) -> tuple[float, float]:
    """Return the logit ``eta*`` of the median point at which the share over points is ``q``.

    The share of right replacements over points at the logit ``eta`` of the median point is
    ``p(eta) = E_b[1 / (1 + e^(-(eta + b)))]``, ``b ~ N(0, sigma2)``. The threshold solves
    ``p(eta*) = q``; with ``sigma2 = 0`` it is ``ln(q / (1 - q))``. The second number is
    ``d eta* / d sigma2 = -(d p / d sigma2) / (d p / d eta)`` with the heat identity
    ``d p / d sigma2 = E_b[f''(eta + b)] / 2``, ``f'' = p (1 - p) (1 - 2 p)``.

    The integral has no closed form, so the passport solves it numerically, by Newton's method from
    :func:`threshold_estimate`, with the quadrature rule of 64 nodes built once.
    """
    if sigma2 <= 0.0:
        # Without a spread the share is the probability itself: the threshold is the logit of
        # the level exactly, as for a fit without the shift.
        return math.log(q / (1.0 - q)), -0.5 * (1.0 - 2.0 * q)
    eta = threshold_estimate(q, sigma2)
    weights = _THRESHOLD_RULE.weights / math.sqrt(math.pi)
    shifts = _SQRT2 * math.sqrt(sigma2) * _THRESHOLD_RULE.nodes
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
