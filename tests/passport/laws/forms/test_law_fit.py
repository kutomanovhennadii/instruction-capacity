"""The fit of a form through points: the weightings, the estimates and the held parameters."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares

from iec.passport.laws.forms import law_fit
from iec.passport.laws.forms.hyperbolic_forms import (
    FractionalHyperbolicForm,
    HyperbolicForm,
    SlantedHyperbolicForm,
)
from iec.passport.laws.forms.law_fit import (
    FREE_METHOD,
    HELD_METHOD,
    covariance_weighting,
    fit_law,
    plain_weighting,
    variance_weighting,
)
from iec.passport.laws.forms.linear_forms import LinearForm

BLOCK = np.array(
    [
        [0.04, 0.01, 0.0, 0.0],
        [0.01, 0.05, 0.01, 0.0],
        [0.0, 0.01, 0.03, 0.005],
        [0.0, 0.0, 0.005, 0.02],
    ]
)


def test_the_weighting_by_the_covariance_is_its_inverse_with_a_root() -> None:
    weighting = covariance_weighting(BLOCK)
    np.testing.assert_allclose(weighting.precision, np.linalg.inv(BLOCK), rtol=1e-10)
    np.testing.assert_allclose(weighting.root @ weighting.root.T, weighting.precision, rtol=1e-10)


def test_a_direction_without_support_has_no_weight() -> None:
    weighting = covariance_weighting(np.array([[1.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(weighting.precision, np.full((2, 2), 0.25), atol=1e-12)
    np.testing.assert_allclose(weighting.precision @ np.array([1.0, -1.0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(weighting.root @ weighting.root.T, weighting.precision, atol=1e-12)


def test_the_weightings_by_the_variances_and_of_plain_least_squares() -> None:
    by_variance = variance_weighting(np.array([4.0, 0.25]))
    assert by_variance.precision.tolist() == [[0.25, 0.0], [0.0, 4.0]]
    assert by_variance.root.tolist() == [[0.5, 0.0], [0.0, 2.0]]
    plain = plain_weighting(3)
    assert plain.precision.tolist() == np.eye(3).tolist()
    assert plain.root.tolist() == np.eye(3).tolist()


def test_a_linear_law_is_generalised_least_squares() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, 2.9, 5.2, 6.8])
    weighting = covariance_weighting(BLOCK)
    fit = fit_law(LinearForm(), x, values, weighting)
    design = np.column_stack([np.ones(4), x])
    estimator = np.linalg.solve(
        design.T @ weighting.precision @ design, design.T @ weighting.precision
    )
    residual = values - design @ (estimator @ values)
    assert fit.converged
    assert fit.form == "linear"
    assert fit.held == ()
    np.testing.assert_allclose(fit.params, estimator @ values, atol=1e-8)
    np.testing.assert_allclose(fit.sensitivity, estimator, atol=1e-10)
    assert fit.chi2 == pytest.approx(float(residual @ weighting.precision @ residual), rel=1e-8)


def test_a_fit_that_starts_at_the_solution_stays_there() -> None:
    form = SlantedHyperbolicForm()
    truth = np.array([2.0, -1.5, 4.0, 0.8])
    x = np.linspace(0.0, 10.0, 12)
    fit = fit_law(form, x, form.value(truth, x), plain_weighting(12), starts=(truth,))
    assert fit.converged
    assert fit.params.tolist() == truth.tolist()
    assert fit.chi2 == 0.0
    assert fit.held == ()


def test_a_parameter_pushed_to_a_bound_is_held_without_sensitivity() -> None:
    x = np.linspace(0.0, 10.0, 11)
    values = 5.0 - 2.0 * x
    fit = fit_law(SlantedHyperbolicForm(), x, values, plain_weighting(11))
    assert fit.converged
    assert fit.held == (2, 3)
    assert fit.params[2] == pytest.approx(0.0, abs=1e-6)
    assert fit.params[3] == pytest.approx(0.001, abs=1e-6)
    assert np.all(fit.sensitivity[2:] == 0.0)
    np.testing.assert_allclose(fit.sensitivity[:2] @ values, fit.params[:2], atol=1e-6)


def test_the_given_bounds_replace_those_of_the_form() -> None:
    form = FractionalHyperbolicForm()
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    values = form.value(np.array([1.0, 4.0, 1.0]), x)
    bounds = (np.array([-np.inf, -np.inf, 5.0]), np.array([np.inf, np.inf, 10.0]))
    fit = fit_law(form, x, values, plain_weighting(5), bounds=bounds)
    assert fit.held == (2,)
    assert fit.params[2] == pytest.approx(5.0, abs=1e-5)


def test_singular_normal_equations_give_no_convergence() -> None:
    fit = fit_law(LinearForm(), np.ones(3), np.array([1.0, 2.0, 3.0]), plain_weighting(3))
    assert not fit.converged
    assert fit.sensitivity.tolist() == np.zeros((2, 3)).tolist()


def test_a_form_without_a_finite_value_at_a_point_does_not_converge() -> None:
    # The hyperbola c + A/x has no value at x = 0: it cannot go through such points.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    found = fit_law(HyperbolicForm(), x, np.array([5.0, 4.0, 3.5, 3.2]), plain_weighting(4))
    assert not found.converged


def test_the_optimizer_follows_the_held_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    methods: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        methods.append(kwargs["method"])
        return scipy_least_squares(*args, **kwargs)

    monkeypatch.setattr(law_fit, "least_squares", spy)
    x = np.linspace(0.0, 10.0, 11)
    fit_law(LinearForm(), x, 1.0 + x, plain_weighting(11))
    fit_law(SlantedHyperbolicForm(), x, 5.0 - 2.0 * x, plain_weighting(11))
    assert (FREE_METHOD, HELD_METHOD) == ("trf", "dogbox")
    # The slanted hyperbola has twelve starts: slopes of both signs.
    assert methods == [FREE_METHOD] + [HELD_METHOD] * 12
