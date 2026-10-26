"""Newton on folded rows: folding is exact, the fit matches the library fit."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.logit_fit import binomial_cluster_covariance, fit_logit, fold_rows

sm = pytest.importorskip("statsmodels.api")

_RUNS = 40
_PER_RUN = 30


def _data(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic data: 40 points of 30 trials, an intercept and indicators of 5 of 6 categories.

    The sixth category is the reference (all indicators zero): otherwise the intercept would
    equal the sum of the indicators, the design would be degenerate, and the library oracle
    would not converge within its iteration limit.
    """
    rng = np.random.default_rng(seed)
    size = _RUNS * _PER_RUN
    x = np.zeros((size, 6))
    x[:, 0] = 1.0
    category = rng.integers(0, 6, size)
    rows = np.flatnonzero(category > 0)
    x[rows, category[rows]] = 1.0
    run = np.repeat(np.arange(_RUNS), _PER_RUN)
    theta = np.array([0.5, -0.4, 0.3, 0.0, -0.8, 0.2])
    y = (rng.random(size) < 1.0 / (1.0 + np.exp(-(x @ theta)))).astype(float)
    weight = rng.uniform(0.5, 1.5, _RUNS)
    return x, y, run, weight


def _reference(
    x: np.ndarray, y: np.ndarray, run: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The library path: a GLM with point weights and the common sandwich on unfolded rows."""
    rows = weight[run]
    fit = sm.GLM(y, x, family=sm.families.Binomial(), var_weights=rows).fit(tol=1e-12)
    mu = 1.0 / (1.0 + np.exp(-(x @ fit.params)))
    variance = rows * mu * (1.0 - mu)
    covariance = binomial_cluster_covariance(
        exog=x, endog=rows * y, groups=run, params=fit.params,
        mean=rows * mu, derivative=variance, variance=variance,
    )
    return np.asarray(fit.params), covariance


def test_folding_keeps_every_trial_and_sorts_by_run() -> None:
    x, y, run, _ = _data()
    rows = fold_rows(x, y, run)
    assert rows.n_trials == len(y)
    assert rows.successes.sum() == pytest.approx(y.sum())
    assert rows.n_runs == _RUNS and np.all(np.diff(rows.run) >= 0)
    assert len(rows.x) < len(x)
    assert rows.starts[0] == 0 and len(rows.starts) == _RUNS
    for index in range(len(rows.x)):
        same = (run == rows.run[index]) & np.all(x == rows.x[index], axis=1)
        assert same.sum() == rows.trials[index]
        assert y[same].sum() == pytest.approx(rows.successes[index])


def test_an_empty_design_folds_to_nothing() -> None:
    rows = fold_rows(np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64))
    assert rows.n_runs == 0 and rows.n_trials == 0 and rows.x.shape == (0, 3)


def test_the_fit_matches_the_library_without_weights() -> None:
    x, y, run, _ = _data()
    ones = np.ones(_RUNS)
    theta, covariance = _reference(x, y, run, ones)
    fit = fit_logit(fold_rows(x, y, run), ones)
    assert fit.converged and not fit.separated
    assert np.max(np.abs(fit.theta - theta)) < 1e-8
    assert np.max(np.abs(fit.covariance - covariance)) / np.max(np.abs(covariance)) < 1e-8


def test_the_fit_matches_the_library_with_run_weights() -> None:
    x, y, run, weight = _data()
    theta, covariance = _reference(x, y, run, weight)
    fit = fit_logit(fold_rows(x, y, run), weight)
    assert np.max(np.abs(fit.theta - theta)) < 1e-8
    assert np.max(np.abs(fit.covariance - covariance)) / np.max(np.abs(covariance)) < 1e-8


def test_a_zero_weight_run_leaves_the_fit_and_the_cluster_count() -> None:
    """A removed point takes part neither in the likelihood nor in the cluster correction."""
    x, y, run, weight = _data()
    rest = run != 0
    theta, covariance = _reference(x[rest], y[rest], run[rest] - 1, weight[1:])
    dropped = weight.copy()
    dropped[0] = 0.0
    fit = fit_logit(fold_rows(x, y, run), dropped)
    assert np.max(np.abs(fit.theta - theta)) < 1e-8
    assert np.max(np.abs(fit.covariance - covariance)) / np.max(np.abs(covariance)) < 1e-8


def test_separated_outcomes_are_flagged_and_count_as_converged() -> None:
    """A column predicts the outcome without errors: separation flag, a converged fit."""
    marker = np.array([1.0] * 10 + [0.0] * 10)
    x = np.column_stack([np.ones(20), marker])
    run = np.arange(20) // 2
    fit = fit_logit(fold_rows(x, marker, run), np.ones(10))
    assert fit.separated is True and fit.converged is True
    assert np.all(np.isfinite(fit.theta))


def test_the_step_limit_reports_non_convergence() -> None:
    x, y, run, weight = _data()
    fit = fit_logit(fold_rows(x, y, run), weight, max_steps=1)
    assert fit.converged is False and fit.steps == 1
