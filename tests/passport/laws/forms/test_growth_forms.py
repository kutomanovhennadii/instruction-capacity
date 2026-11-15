"""The forms of a growing loss: the exponential, the binary exponential, saturation, power."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from iec.passport.laws.forms.growth_forms import (
    BinaryExponentialForm,
    ExponentialForm,
    PowerForm,
    SaturationForm,
)
from iec.passport.laws.forms.law_form import LawForm


def test_the_exponential_law_falls_from_c_plus_a_to_c() -> None:
    start, half, end = ExponentialForm().value(
        np.array([1.0, 2.0, 0.5]), np.array([0.0, 0.5 * math.log(2.0), 1e3])
    )
    assert start == 3.0
    assert half == pytest.approx(2.0)
    assert end == pytest.approx(1.0)


def test_the_binary_exponential_grows_its_loss_by_two_to_gamma_per_unit() -> None:
    values = BinaryExponentialForm().value(np.array([5.0, 1.0, 1.0]), np.array([0.0, 1.0, 2.0]))
    assert values.tolist() == [4.0, 3.0, 1.0]


def test_the_saturation_starts_at_mu_and_loses_at_most_a() -> None:
    start, middle, end = SaturationForm().value(
        np.array([1.0, 0.4, 2.0]), np.array([0.0, 2.0, 1e3])
    )
    assert start == 1.0
    assert middle == pytest.approx(1.0 - 0.4 * (1.0 - math.exp(-1.0)))
    assert end == pytest.approx(0.6)


def test_the_power_law_is_lambda_at_one() -> None:
    values = PowerForm().value(np.array([2.0, 0.5, 2.0]), np.array([1.0, 3.0]))
    assert values.tolist() == [2.0, -2.0]


def test_only_the_binary_exponential_has_a_limit() -> None:
    assert BinaryExponentialForm().limit(np.array([5.0, 1.0, 1.0])) == 5.0
    assert ExponentialForm().limit(np.array([1.0, 2.0, 0.5])) is None
    assert SaturationForm().limit(np.array([1.0, 0.4, 2.0])) is None
    assert PowerForm().limit(np.array([2.0, 0.5, 2.0])) is None


def test_the_formulas_name_the_symbol() -> None:
    assert ExponentialForm().formula("q") == "c + A·e^(−q/τ)"
    assert BinaryExponentialForm().formula("ℓ") == "μ − A·2^(γ·ℓ)"
    assert SaturationForm().formula("p") == "μ − A·(1 − e^(−p/p_c))"
    assert PowerForm().formula("r") == "λ − A·(r^γ − 1)"


def test_the_ranges() -> None:
    x = np.array([0.0, 1.0])
    lengths = [[-math.inf, 0.0, 0.001], [math.inf, math.inf, 1000.0]]
    assert [side.tolist() for side in ExponentialForm().bounds(x)] == lengths
    assert [side.tolist() for side in SaturationForm().bounds(x)] == lengths
    assert [side.tolist() for side in BinaryExponentialForm().bounds(x)] == [
        [-math.inf, 0.0, 0.005],
        [math.inf, math.inf, 2.0],
    ]
    assert [side.tolist() for side in PowerForm().bounds(x)] == [
        [-math.inf, 0.0, 0.001],
        [math.inf, math.inf, 10.0],
    ]


def test_a_single_abscissa_keeps_a_positive_range_of_the_length() -> None:
    lower, upper = ExponentialForm().bounds(np.array([3.0, 3.0]))
    assert lower[2] == pytest.approx(1e-9)
    assert upper[2] == pytest.approx(1e-3)


def test_the_starts() -> None:
    x = np.array([0.0, 0.5, 1.0])
    exponential = ExponentialForm().starts(x, np.array([3.0, 2.0, 1.5]))
    binary = BinaryExponentialForm().starts(x, np.array([4.0, 3.0, 1.0]))
    saturation = SaturationForm().starts(x, np.array([1.0, 0.75, 0.5]))
    power = PowerForm().starts(np.array([1.0, 2.0, 5.0]), np.array([2.0, 1.0, 0.0]))
    assert [start.tolist() for start in exponential] == [[1.5, 1.5, 1.0 / 3.0]]
    assert [start.tolist() for start in binary] == [[4.0, 1.0, 0.5]]
    assert [start.tolist() for start in saturation] == [[1.0, 0.5, 0.5]]
    assert [start.tolist() for start in power] == [[2.0, 0.5, 1.0]]


@pytest.mark.parametrize(
    ("form", "params", "x"),
    [
        (ExponentialForm(), [1.0, 2.0, 0.5], [0.0, 0.3, 1.2]),
        (BinaryExponentialForm(), [5.0, 0.8, 0.3], [-2.0, 0.0, 3.5]),
        (SaturationForm(), [1.0, 0.4, 2.0], [0.0, 1.0, 6.0]),
        (PowerForm(), [2.0, 0.5, 1.5], [1.0, 2.0, 5.0]),
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
    assert ExponentialForm().held_candidates == (1, 2)
    assert BinaryExponentialForm().held_candidates == (2,)
    assert SaturationForm().held_candidates == (2,)
    assert PowerForm().held_candidates == (2,)
    assert ExponentialForm().parameters == ("c", "A", "tau")
    assert BinaryExponentialForm().parameters == ("mu", "A", "gamma")
    assert SaturationForm().parameters == ("mu", "A", "x_c")
    assert PowerForm().parameters == ("lambda", "A", "gamma")
