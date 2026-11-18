"""The hyperbolic forms: the hyperbola, the fractional, the slanted, the two-slope and the two-asymptote one."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from iec.passport.laws.forms.hyperbolic_forms import (
    AsymptoticHyperbolicForm,
    FractionalHyperbolicForm,
    HyperbolicForm,
    SlantedHyperbolicForm,
    TwoSlopeHyperbolicForm,
)
from iec.passport.laws.forms.law_form import LawForm


def test_the_hyperbola_falls_to_c() -> None:
    values = HyperbolicForm().value(np.array([1.0, 4.0]), np.array([1.0, 2.0, 4.0]))
    assert values.tolist() == [5.0, 3.0, 2.0]


def test_the_fractional_hyperbola_keeps_half_of_its_fall_at_x0() -> None:
    values = FractionalHyperbolicForm().value(
        np.array([1.0, 4.0, 10.0]), np.array([0.0, 10.0, 30.0])
    )
    assert values.tolist() == [5.0, 3.0, 2.0]


def test_the_slanted_hyperbola_tends_to_its_two_asymptotes() -> None:
    params = np.array([2.0, -1.0, 5.0, 0.5])
    left, corner, right = SlantedHyperbolicForm().value(params, np.array([-1e4, 5.0, 1e4]))
    assert left == pytest.approx(2.0, abs=1e-4)
    assert corner == 1.75
    assert right == pytest.approx(2.0 - (1e4 - 5.0), abs=1e-4)


def test_the_two_slope_hyperbola_tends_to_its_two_slanted_asymptotes() -> None:
    params = np.array([2.0, -1.5, 5.0, 0.5, 0.4])
    left, right = TwoSlopeHyperbolicForm().value(params, np.array([-1e4, 1e4]))
    assert left == pytest.approx(2.0 + 0.4 * (-1e4 - 5.0), abs=1e-4)
    assert right == pytest.approx(2.0 + (0.4 - 1.5) * (1e4 - 5.0), abs=1e-4)


def test_the_two_slope_hyperbola_without_a_left_slope_is_the_slanted_one() -> None:
    x = np.linspace(-3.0, 12.0, 31)
    slanted = SlantedHyperbolicForm().value(np.array([2.0, -1.5, 5.0, 0.5]), x)
    two = TwoSlopeHyperbolicForm().value(np.array([2.0, -1.5, 5.0, 0.5, 0.0]), x)
    np.testing.assert_allclose(two, slanted, rtol=1e-12, atol=1e-12)


def test_the_two_asymptote_hyperbola_is_the_slanted_one_written_by_its_line() -> None:
    x = np.linspace(-3.0, 8.0, 23)
    kappa, intercept, omega, mu = -2.0, 6.0, 0.5, 1.0
    corner = (mu - intercept) / kappa
    two = AsymptoticHyperbolicForm().value(np.array([kappa, intercept, omega, mu]), x)
    slanted = SlantedHyperbolicForm().value(np.array([mu, kappa, corner, omega]), x)
    np.testing.assert_allclose(two, slanted, rtol=1e-12, atol=1e-12)


def test_the_slanted_forms_have_the_limit_mu_and_the_others_none() -> None:
    assert SlantedHyperbolicForm().limit(np.array([2.0, -1.0, 5.0, 0.5])) == 2.0
    assert AsymptoticHyperbolicForm().limit(np.array([-2.0, 6.0, 0.5, 1.0])) == 1.0
    assert TwoSlopeHyperbolicForm().limit(np.array([2.0, -1.5, 5.0, 0.5, 0.4])) is None
    assert HyperbolicForm().limit(np.array([1.0, 4.0])) is None
    assert FractionalHyperbolicForm().limit(np.array([1.0, 4.0, 10.0])) is None


def test_the_formulas_name_the_symbol() -> None:
    assert HyperbolicForm().formula("v") == "c + A/v"
    assert FractionalHyperbolicForm().formula("v") == "c + A/(1 + v/v₀)"
    assert SlantedHyperbolicForm().formula("u") == "μ + (κ/2)·[(u − u*) + √((u − u*)² + ω²)]"
    two_slope = "μ + β·(u − u*) + (κ/2)·[(u − u*) + √((u − u*)² + ω²)]"
    assert TwoSlopeHyperbolicForm().formula("u") == two_slope
    expected = "[(κ·ℓ + λ) + μ − √((κ·ℓ + λ − μ)² + κ²·ω²)]/2"
    assert AsymptoticHyperbolicForm().formula("ℓ") == expected


def test_the_range_of_the_slanted_hyperbola_lies_inside_its_axis() -> None:
    # The slope of the slanted asymptote takes either sign.
    lower, upper = SlantedHyperbolicForm().bounds(np.array([0.0, 2.5, 10.0]))
    assert lower.tolist() == [-math.inf, -math.inf, 0.0, 0.001]
    assert upper.tolist() == [math.inf, math.inf, 10.0, 10.0]
    _, narrow = SlantedHyperbolicForm().bounds(np.array([0.0, 0.001]))
    assert narrow.tolist() == [math.inf, math.inf, 0.002, 0.002]


def test_the_range_of_the_two_slope_hyperbola_frees_both_slopes() -> None:
    lower, upper = TwoSlopeHyperbolicForm().bounds(np.array([0.0, 2.5, 10.0]))
    assert lower.tolist() == [-math.inf, -math.inf, 0.0, 0.001, -math.inf]
    assert upper.tolist() == [math.inf, math.inf, 10.0, 10.0, math.inf]


def test_the_starts_of_the_two_slope_hyperbola_follow_two_lines_then_the_slanted_ones() -> None:
    x, values = np.array([0.0, 2.0, 4.0, 6.0, 8.0]), np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    starts = TwoSlopeHyperbolicForm().starts(x, values)
    assert len(starts) == 9
    # The lines to the left and to the right of the corners 2, 4 and 6 of the hill.
    np.testing.assert_allclose(
        np.array(starts[:3]),
        [[1.0, -0.7, 2.0, 0.5, 0.5], [2.0, -1.0, 4.0, 0.5, 0.5], [1.6, -0.7, 6.0, 0.5, 0.2]],
        atol=1e-12,
    )
    assert [start.tolist() for start in starts[3:]] == [
        [0.0, slope, corner, 0.5, 0.0] for slope in (-1.0, -3.0) for corner in (2.0, 4.0, 6.0)
    ]


def test_the_starts_of_the_slanted_hyperbola_take_both_signs_of_the_slope() -> None:
    starts = SlantedHyperbolicForm().starts(np.array([0.0, 4.0, 8.0]), np.array([3.0, 2.0, 1.0]))
    assert [start.tolist() for start in starts] == [
        [3.0, slope, corner, 0.5]
        for slope in (-1.0, -3.0, 1.0, 3.0)
        for corner in (2.0, 4.0, 6.0)
    ]


def test_the_ranges_of_the_fractional_and_the_two_asymptote_hyperbola() -> None:
    lower, upper = FractionalHyperbolicForm().bounds(np.array([2.0, 50.0]))
    assert lower.tolist() == [-math.inf, -math.inf, 0.002]
    assert upper.tolist() == [math.inf, math.inf, 50000.0]
    lower, upper = AsymptoticHyperbolicForm().bounds(np.array([0.0, 3.0]))
    assert lower.tolist() == [-math.inf, -math.inf, 0.001, -math.inf]
    assert upper.tolist() == [0.0, math.inf, 3.0, math.inf]


def test_the_starts_of_the_other_hyperbolas() -> None:
    x, values = np.array([1.0, 2.0, 4.0]), np.array([5.0, 3.0, 2.0])
    hyperbola = HyperbolicForm().starts(x, values)
    fractional = FractionalHyperbolicForm().starts(x, values)
    two = AsymptoticHyperbolicForm().starts(x, values)
    assert [start.tolist() for start in hyperbola] == [[2.0, 3.0]]
    assert [start.tolist() for start in fractional] == [[2.0, 3.0, 1.0]]
    assert [start.tolist() for start in two] == [[-2.0, 0.0, 1.0, 5.0]]


@pytest.mark.parametrize(
    ("form", "params", "x"),
    [
        (HyperbolicForm(), [1.0, 4.0], [0.5, 2.0, 7.0]),
        (FractionalHyperbolicForm(), [1.0, 4.0, 10.0], [0.0, 3.0, 40.0]),
        (SlantedHyperbolicForm(), [2.0, -1.5, 4.0, 0.8], [0.0, 3.5, 4.0, 9.0]),
        (TwoSlopeHyperbolicForm(), [2.0, -1.5, 4.0, 0.8, 0.3], [0.0, 3.5, 4.0, 9.0]),
        (AsymptoticHyperbolicForm(), [-2.0, 6.0, 0.5, 1.0], [-1.0, 2.5, 6.0]),
    ],
)
def test_the_derivatives_are_those_of_the_value(
    numeric_jacobian: Callable[..., np.ndarray],
    form: LawForm,
    params: list[float],
    x: list[float],
) -> None:
    point, abscissae = np.array(params), np.array(x)
    np.testing.assert_allclose(
        form.jacobian(point, abscissae),
        numeric_jacobian(form, point, abscissae),
        rtol=1e-6,
        atol=1e-7,
    )


def test_the_parameters_that_may_be_held_and_their_names() -> None:
    assert HyperbolicForm().held_candidates == ()
    assert FractionalHyperbolicForm().held_candidates == (2,)
    assert SlantedHyperbolicForm().held_candidates == (2, 3)
    assert TwoSlopeHyperbolicForm().held_candidates == (2, 3)
    assert AsymptoticHyperbolicForm().held_candidates == ()
    assert HyperbolicForm().parameters == ("c", "A")
    assert FractionalHyperbolicForm().parameters == ("c", "A", "x0")
    assert SlantedHyperbolicForm().parameters == ("mu", "kappa", "corner", "omega")
    assert TwoSlopeHyperbolicForm().parameters == ("mu", "kappa", "corner", "omega", "beta")
    assert AsymptoticHyperbolicForm().parameters == ("kappa", "lambda", "omega", "mu")
