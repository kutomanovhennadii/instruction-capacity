"""The continuous IEC of the second stage: the cumulative price and its derivatives."""

from __future__ import annotations

import math

import pytest

from iec.passport.second_stage_laws import _cumulative_cost, _cumulative_derivatives

#: A point of the law: the number of rules m, the slope kappa_M per doubling of the table, the
#: scale A_R and the exponent gamma.
_POINT = (12.5, -0.3, 0.06, 0.8)


def _rule_price(r: float, kappa_m: float, amplitude: float, gamma: float) -> float:
    """The price of rule ``r``: the table-size discount and the late-rule degradation."""
    return -kappa_m / (r * math.log(2.0)) + amplitude * (r**gamma - 1.0)


def test_the_cumulative_price_is_the_integral_of_the_rule_price() -> None:
    m, kappa_m, amplitude, gamma = _POINT
    steps = 20000
    width = (m - 1.0) / steps
    integral = width * sum(
        _rule_price(1.0 + (k + 0.5) * width, kappa_m, amplitude, gamma) for k in range(steps)
    )
    assert _cumulative_cost(m, kappa_m, amplitude, gamma) == pytest.approx(integral, rel=1e-6)


def test_the_derivatives_match_finite_differences() -> None:
    derivatives = _cumulative_derivatives(*_POINT)
    step = 1e-6
    for slot in range(len(_POINT)):
        up = list(_POINT)
        down = list(_POINT)
        up[slot] += step
        down[slot] -= step
        numeric = (_cumulative_cost(*up) - _cumulative_cost(*down)) / (2.0 * step)
        assert derivatives[slot] == pytest.approx(numeric, rel=1e-6, abs=1e-9)
