"""The random-intercept logit: the plain logit without a spread, an independent integral,
recovery of a known shift, warm starts, zero weights, the gradient with the nodes placed anew,
the threshold of a level."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from iec.passport.fitting.acceleration.folded_rows import FoldedRows, fold_rows
from iec.passport.fitting.acceleration.quadrature_rule import build_quadrature_rule
from iec.passport.fitting.acceleration.sparse_design import build_sparse_design
from iec.passport.fitting.glmm_fit import (
    GlmmFitter,
    _evaluate,
    _modes,
    _node_terms,
    _value,
    fit_glmm,
    fit_spread,
    marginal_threshold,
    threshold_estimate,
)
from iec.passport.fitting.logit_fit import fit_logit, logistic_array


def _rows(theta: np.ndarray, sigma2: float, runs: int, per_run: int, seed: int) -> FoldedRows:
    """Trials of a level and a slope over ``t``; every point has its shift ``N(0, sigma2)``."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-2.0, 2.0, size=runs * per_run)
    x = np.column_stack([np.ones_like(t), t])
    run = np.repeat(np.arange(runs), per_run)
    shift = rng.normal(0.0, math.sqrt(sigma2), runs) if sigma2 > 0.0 else np.zeros(runs)
    p = 1.0 / (1.0 + np.exp(-(x @ theta + shift[run])))
    outcomes = (rng.random(len(t)) < p).astype(float)
    return fold_rows(x, outcomes, run)


def _start(rows: FoldedRows, weight: np.ndarray) -> np.ndarray:
    return fit_logit(rows, weight).theta


def test_without_a_spread_the_fit_is_the_plain_logit() -> None:
    rows = _rows(np.array([0.3, 0.8]), 0.0, runs=20, per_run=30, seed=5)
    weight = np.ones(rows.n_runs)

    fit = fit_glmm(rows, weight, _start(rows, weight), 1.0)

    assert fit.converged and fit.on_floor
    assert fit.theta == pytest.approx(_start(rows, weight), abs=1e-6)
    assert np.all(fit.covariance[-1] == 0.0) and np.all(fit.covariance[:, -1] == 0.0)


def _marginal_loglik(rows: FoldedRows, theta: np.ndarray, sigma2: float) -> float:
    """The marginal log-likelihood by the trapezoid rule on a fine grid of the shift."""
    sigma = math.sqrt(sigma2)
    grid = np.linspace(-10.0 * sigma, 10.0 * sigma, 1201)
    eta = (rows.x @ theta)[:, None] + grid[None, :]
    row_loglik = rows.successes[:, None] * eta - rows.trials[:, None] * np.logaddexp(0.0, eta)
    per_point = np.add.reduceat(row_loglik, rows.starts, axis=0)
    per_point += -grid[None, :] ** 2 / (2.0 * sigma2) - math.log(sigma) - 0.5 * math.log(
        2.0 * math.pi
    )
    top = per_point.max(axis=1, keepdims=True)
    integral = np.trapezoid(np.exp(per_point - top), grid, axis=1)
    return float(np.sum(top[:, 0] + np.log(integral)))


def test_the_fit_matches_an_independent_integral() -> None:
    """The maximum of the trapezoid integral, found by a general optimiser, is the fit."""
    rows = _rows(np.array([0.5, -0.7]), 1.0, runs=30, per_run=20, seed=11)
    weight = np.ones(rows.n_runs)
    fit = fit_glmm(rows, weight, _start(rows, weight), 1.0)

    def minus(params: np.ndarray) -> float:
        return -_marginal_loglik(rows, params[:2], math.exp(2.0 * params[2]))

    best = minimize(
        minus,
        np.r_[_start(rows, weight), 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000},
    )

    assert fit.converged
    assert fit.theta == pytest.approx(best.x[:2], abs=2e-3)
    assert fit.sigma2 == pytest.approx(math.exp(2.0 * best.x[2]), rel=1e-2)
    assert fit.loglik == pytest.approx(-best.fun, abs=1e-3)


def test_a_known_shift_is_recovered_within_its_errors() -> None:
    truth = np.array([0.4, -0.9])
    rows = _rows(truth, 1.5, runs=300, per_run=40, seed=17)
    weight = np.ones(rows.n_runs)

    fit = fit_glmm(rows, weight, _start(rows, weight), 1.0)

    se = np.sqrt(np.diag(fit.covariance))
    assert fit.converged and not fit.on_floor
    assert np.all(np.abs(fit.theta - truth) < 3.0 * se[:2])
    assert abs(fit.sigma2 - 1.5) < 3.0 * se[2]


def test_a_warm_start_reaches_the_estimates_of_a_cold_one() -> None:
    rows = _rows(np.array([0.4, -0.9]), 1.5, runs=200, per_run=30, seed=23)
    weight = np.ones(rows.n_runs)
    shorter = weight.copy()
    shorter[180:] = 0.0
    cold = fit_glmm(rows, weight, _start(rows, weight), 1.0)
    partial = fit_glmm(rows, shorter, _start(rows, shorter), 1.0)

    warm = fit_glmm(rows, weight, partial.theta, partial.sigma2, partial.modes)

    assert warm.converged and warm.steps < cold.steps
    assert warm.theta == pytest.approx(cold.theta, abs=1e-7)
    assert warm.sigma2 == pytest.approx(cold.sigma2, rel=1e-7)


def test_a_zero_weight_removes_the_point() -> None:
    rows = _rows(np.array([0.4, -0.9]), 1.5, runs=60, per_run=25, seed=29)
    weight = np.ones(rows.n_runs)
    weight[:10] = 0.0
    with_zeros = fit_glmm(rows, weight, _start(rows, weight), 1.0)
    kept = rows.run >= 10
    reduced = FoldedRows(
        x=rows.x[kept],
        successes=rows.successes[kept],
        trials=rows.trials[kept],
        run=rows.run[kept] - 10,
        starts=rows.starts[10:] - rows.starts[10],
        n_runs=rows.n_runs - 10,
    )
    ones = np.ones(reduced.n_runs)

    without = fit_glmm(reduced, ones, _start(reduced, ones), 1.0)

    assert with_zeros.theta == pytest.approx(without.theta, abs=1e-8)
    assert with_zeros.sigma2 == pytest.approx(without.sigma2, rel=1e-8)
    assert with_zeros.covariance == pytest.approx(without.covariance, rel=1e-6, abs=1e-12)


def test_the_gradient_is_the_derivative_of_the_value_with_the_nodes_placed_anew() -> None:
    """The step goes by the derivative of the value the line search compares: the nodes of every
    point move with the parameters. Five nodes make the move of the nodes visible."""
    rows = _rows(np.array([0.4, -0.9]), 1.5, runs=60, per_run=25, seed=37)
    design = build_sparse_design(rows)
    weight = np.ones(rows.n_runs)
    spread_x = np.column_stack([np.ones(rows.n_runs), np.arange(rows.n_runs) % 2])
    rule = build_quadrature_rule(5)
    point = np.array([0.2, -0.5, 0.3, -0.2])

    def placed(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sigma2 = np.exp(2.0 * (spread_x @ params[2:]))
        return _modes(design, design.x @ params[:2], sigma2, np.zeros(rows.n_runs))

    def value(params: np.ndarray) -> float:
        mode, scale = placed(params)
        return _value(design, weight, params[:2], spread_x @ params[2:], mode, scale, rule)[0]

    mode, scale = placed(point)
    log_sigma = spread_x @ point[2:]
    gradient = _evaluate(design, weight, point[:2], log_sigma, spread_x, mode, scale, rule).gradient
    step = 1e-5
    numeric = np.array(
        [(value(point + step * e) - value(point - step * e)) / (2.0 * step) for e in np.eye(4)]
    )
    _, b, eta, posterior = _value(design, weight, point[:2], log_sigma, mode, scale, rule)
    residual = rows.successes[:, None] - rows.trials[:, None] * logistic_array(eta)
    moved = _node_terms(
        design, point[:2], log_sigma, spread_x, mode, scale, rule, b, residual, posterior
    ).T @ weight

    assert gradient == pytest.approx(numeric, rel=1e-6, abs=1e-6)
    assert float(np.max(np.abs(moved))) > 1e-3


@pytest.mark.parametrize("q", [0.95, 0.5, 0.2])
@pytest.mark.parametrize("sigma2", [0.0, 1.0, 6.8])
def test_the_threshold_gives_the_share_over_points(q: float, sigma2: float) -> None:
    eta, slope = marginal_threshold(q, sigma2)

    if sigma2 == 0.0:
        assert eta == pytest.approx(math.log(q / (1.0 - q)), abs=1e-12)
    else:
        sigma = math.sqrt(sigma2)
        grid = np.linspace(-12.0 * sigma, 12.0 * sigma, 20001)
        density = np.exp(-grid**2 / (2.0 * sigma2)) / math.sqrt(2.0 * math.pi * sigma2)
        share = np.trapezoid(density / (1.0 + np.exp(-(eta + grid))), grid)
        # 64 Gauss-Hermite nodes miss the share by 5e-9 at sigma2 = 6.8: 3e-8 of a logit.
        assert share == pytest.approx(q, abs=1e-7)
    if q == 0.5:
        assert eta == pytest.approx(0.0, abs=1e-12) and slope == pytest.approx(0.0, abs=1e-12)
    step = 1e-5
    numeric = (
        marginal_threshold(q, sigma2 + step)[0] - marginal_threshold(q, max(sigma2 - step, 0.0))[0]
    ) / (step if sigma2 == 0.0 else 2.0 * step)
    assert slope == pytest.approx(numeric, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize(
    ("sigma2", "threshold"),
    [(1.73, 3.655), (2.27, 3.852), (5.90, 4.973), (13.66, 6.767), (50.0, 12.017)],
)
def test_the_threshold_of_the_level_095_keeps_its_value(sigma2: float, threshold: float) -> None:
    assert marginal_threshold(0.95, sigma2)[0] == pytest.approx(threshold, abs=5.01e-4)


@pytest.mark.parametrize("sigma2", [0.5, 1.0, 1.73, 2.27, 5.9, 13.66, 30.0, 50.0])
def test_the_estimate_lies_within_its_promise_at_the_level_095(sigma2: float) -> None:
    exact = marginal_threshold(0.95, sigma2)[0]
    assert abs(threshold_estimate(0.95, sigma2) - exact) <= 0.014 * exact


def test_the_estimate_without_a_spread_is_the_logit_of_the_level() -> None:
    assert threshold_estimate(0.95, 0.0) == pytest.approx(math.log(19.0), abs=1e-12)
    assert threshold_estimate(0.95, -1.0) == pytest.approx(math.log(19.0), abs=1e-12)
    assert threshold_estimate(0.5, 13.0) == 0.0


def test_one_column_of_the_spread_is_the_fit_with_one_variance() -> None:
    rows = _rows(np.array([1.0, -0.5]), 1.5, 120, 30, 3)
    weight = np.ones(rows.n_runs)
    start = _start(rows, weight)
    plain = fit_glmm(rows, weight, start, 1.0)
    spread = fit_spread(rows, weight, start, np.array([0.0]), np.ones((rows.n_runs, 1)))
    assert plain.converged and spread.converged
    assert spread.theta == pytest.approx(plain.theta, abs=1e-8)
    assert spread.gamma[0] == pytest.approx(0.5 * math.log(plain.sigma2), abs=1e-8)
    assert spread.loglik == pytest.approx(plain.loglik, abs=1e-6)
    size = len(plain.theta)
    assert spread.covariance[:size, :size] == pytest.approx(plain.covariance[:size, :size], abs=1e-8)
    assert spread.steps == plain.steps


def test_a_fitter_from_several_starts_gives_the_fits_of_one_call_each() -> None:
    rows = _rows(np.array([0.4, -0.9]), 1.5, runs=80, per_run=20, seed=31)
    weight = np.ones(rows.n_runs)
    start = _start(rows, weight)
    fitter = GlmmFitter(build_sparse_design(rows))

    for sigma2_start in (1.0, 4.0):
        reused = fitter.fit(weight, start, sigma2_start)
        single = fit_glmm(rows, weight, start, sigma2_start)
        assert np.array_equal(reused.theta, single.theta)
        assert reused.sigma2 == single.sigma2
        assert np.array_equal(reused.covariance, single.covariance)
        assert reused.steps == single.steps


def test_the_spread_of_a_fitter_is_the_spread_of_one_call() -> None:
    rows = _rows(np.array([1.0, -0.5]), 1.5, 120, 30, 3)
    weight = np.ones(rows.n_runs)
    start = _start(rows, weight)
    spread_x = np.column_stack([np.ones(rows.n_runs), np.arange(rows.n_runs) % 2])
    gamma = np.array([0.0, 0.0])

    reused = GlmmFitter(build_sparse_design(rows)).fit_spread(weight, start, gamma, spread_x)
    single = fit_spread(rows, weight, start, gamma, spread_x)

    assert np.array_equal(reused.theta, single.theta)
    assert np.array_equal(reused.gamma, single.gamma)
    assert np.array_equal(reused.covariance, single.covariance)
    assert reused.steps == single.steps
