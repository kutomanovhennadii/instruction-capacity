"""The random-intercept logit: the plain logit without a spread, an independent integral,
recovery of a known shift, warm starts, zero weights, the threshold of a level."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from iec.passport.glmm_fit import fit_glmm, marginal_threshold
from iec.passport.logit_fit import FoldedRows, fit_logit, fold_rows


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
