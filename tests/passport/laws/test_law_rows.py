"""Tools of the laws: the joint covariance, half-widths, own errors, a capacity."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from iec.passport.fitting.chain.fit_paths import METHOD_GLMM, METHOD_LOGIT
from iec.passport.fitting.glmm_fit import marginal_threshold
from iec.passport.fitting.load_spread import LoadSpread, SpreadAxis
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.law_rows import (
    HALF,
    LEVEL,
    LEVEL_THRESHOLD,
    capacity_row,
    fit_summary,
    half_width,
    joint_covariance,
    own_variances,
)


def _fit(method: str, sigma2_row: tuple[float, ...]) -> VariableFit:
    return VariableFit(
        variable="F",
        method=method,
        names=("a", "b"),
        theta=(1.0, 2.0),
        se=(0.1, 0.2),
        covariance=((0.01, 0.0), (0.0, 0.04)),
        dropped=("c",),
        sigma2=2.0 if sigma2_row else None,
        sigma2_covariance=sigma2_row,
        steps=3,
        rows=4,
        points=5,
        failures=6,
        trials=7,
    )


def test_joint_covariance_keeps_the_row_of_sigma2() -> None:
    joint = joint_covariance(_fit(METHOD_GLMM, (0.001, 0.002, 0.25)))
    assert joint[2, 2] == 0.25
    assert joint[0, 2] == joint[2, 0] == 0.001


def test_joint_covariance_without_the_shift_has_a_zero_row() -> None:
    joint = joint_covariance(_fit(METHOD_LOGIT, ()))
    assert joint.shape == (3, 3)
    assert not joint[2].any()


def test_half_width_pads_a_short_gradient() -> None:
    joint = np.diag([0.01, 0.04, 0.25])
    assert half_width(np.array([1.0, 0.0]), joint) == pytest.approx(HALF * 0.1)
    assert half_width(np.array([0.0, 0.0, 1.0]), joint) == pytest.approx(HALF * 0.5)


def test_own_variances_give_the_variances_of_differences() -> None:
    # Bins with independent errors 0.04 and 0.09 and a reference bin with an error 0.01: the
    # coefficients count from the reference bin, so its variance is common to both.
    block = np.array([[0.05, 0.01], [0.01, 0.10]])
    q = own_variances(block, implicit_reference=True)
    assert q == pytest.approx([0.01, 0.04, 0.09], rel=1e-4)


def test_own_variances_without_a_reference_bin() -> None:
    block = np.array([[1.04, 1.0, 1.0], [1.0, 1.09, 1.0], [1.0, 1.0, 1.16]])
    q = own_variances(block, implicit_reference=False)
    assert q == pytest.approx([0.04, 0.09, 0.16], rel=1e-4)


def test_fit_summary_gives_the_half_width_of_sigma2() -> None:
    summary = fit_summary(_fit(METHOD_GLMM, (0.0, 0.0, 0.25)))
    assert summary["sigma2_half"] == pytest.approx(HALF * 0.5)
    assert summary["dropped"] == ["c"]
    assert summary["parameters"] == 2


def test_capacity_row_marks_a_value_past_the_measured_region() -> None:
    inside = capacity_row("x", 0.95, 3.0, 1.0, 10.0)
    assert inside["value"] == 8.0
    assert inside["interval"] == [4.0, 16.0]
    assert inside["plus_minus"] == pytest.approx(math.log(2.0) * 8.0)
    assert inside["status"] == "measured"
    assert capacity_row("x", 0.95, 4.0, 1.0, 10.0)["status"] == "extrapolated"
    assert capacity_row("x", 0.95, 1.0, 1.0, 100.0, scale=20.0)["value"] == 40.0


def test_capacity_row_holds_a_number_past_the_largest_float_as_none() -> None:
    # A nearly flat law: the capacity holds in a float, the upper bound of its interval does not.
    wide = capacity_row("x", 0.5, 30.0, 1000.0, 10.0)
    assert wide["value"] == 2.0**30
    assert wide["interval"] == [2.0**-970, None]
    assert wide["plus_minus"] == pytest.approx(math.log(2.0) * 2.0**30 * 1000.0)
    assert wide["status"] == "extrapolated"
    # The capacity itself past the largest float, by the power and by its unit.
    past = capacity_row("x", 0.5, 1100.0, 1.0, 10.0)
    assert past["value"] is None
    assert past["interval"] == [None, None]
    assert past["plus_minus"] is None
    assert past["status"] == "extrapolated"
    assert capacity_row("x", 0.5, 1023.0, 0.0, 10.0, scale=20.0)["value"] is None


@pytest.mark.parametrize("sigma2", [0.0, 0.5, 4.0, 16.0, 64.0])
def test_the_threshold_of_the_level_of_the_capacities_is_zero_at_any_variance(sigma2: float) -> None:
    assert LEVEL == 0.5
    assert marginal_threshold(LEVEL, sigma2)[0] == pytest.approx(LEVEL_THRESHOLD, abs=1e-12)


_SINGLE = SpreadAxis(((1,),), (None,), (1.0,), (1.0,), (40,))
_LENGTH = SpreadAxis(((1,), (2,)), (None, 1), (100.0, 4000.0), (0.5, 0.5), (40, 40))


def _spread_fit() -> VariableFit:
    fit = _fit(METHOD_GLMM, (0.0, 0.0, 0.25))
    return replace(
        fit,
        method="glmm_heterogeneous",
        spread=LoadSpread((0.3, 0.5), (_LENGTH, _SINGLE, _SINGLE)),
        spread_covariance=tuple(tuple(row) for row in (np.eye(len(fit.names) + 2) * 1e-4).tolist()),
    )


def test_the_summary_gives_the_variance_of_a_group() -> None:
    fit = _spread_fit()
    summary = fit_summary(replace(fit, loglik=10.0, loglik_common=4.0))
    assert summary["spread"]["loglik_gain"] == pytest.approx(12.0)
    assert summary["spread"]["parameters"] == 1
    assert summary["spread"]["V"][0]["sigma2"] == pytest.approx(math.exp(0.6))
    assert summary["spread"]["V"][1]["sigma2"] == pytest.approx(math.exp(1.6))
    assert summary["spread"]["V"][1]["half"] > 0.0
