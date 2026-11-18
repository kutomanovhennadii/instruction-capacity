"""The capacity by the root of the laws: the root, laws without a root, the ceiling, the error."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iec.passport.laws.block_choice import LawStack
from iec.passport.quantities.steps.law_capacity import odds_factor_row, root_capacity_row

#: A law ``level + slope·x`` of two parameters, one coefficient each.
STACK = LawStack({"law": (np.array([3.0, -1.0]), np.eye(2))})


def _root(vector: np.ndarray) -> float:
    """Return the root of ``level + slope·x = 0``; ``nan`` - a law that does not fall."""
    level, slope = STACK.part(vector, "law")
    return math.nan if slope >= 0.0 else float(-level / slope)


def test_a_falling_law_gives_its_root() -> None:
    row = root_capacity_row("instruction_capacity", _root, STACK, np.zeros((3, 3)), 16.0)
    assert row["value"] == pytest.approx(8.0)
    assert row["interval"] == pytest.approx([8.0, 8.0])
    assert row["status"] == "measured"


def test_the_error_of_the_root_goes_by_the_delta_method() -> None:
    joint = np.diag([0.01, 0.0001, 0.0])
    row = root_capacity_row("instruction_capacity", _root, STACK, joint, 16.0)
    # x = 3: dx/dlevel = 1, dx/dslope = 3; the half-width of log2 is 1.96·sqrt(0.01 + 9·0.0001).
    half = 1.96 * math.sqrt(0.01 + 9.0 * 0.0001)
    assert row["interval"] == pytest.approx([2.0 ** (3.0 - half), 2.0 ** (3.0 + half)])


def test_a_law_that_does_not_fall_gives_no_root() -> None:
    flat = LawStack({"law": (np.array([3.0, 0.0]), np.eye(2))})
    row = root_capacity_row(
        "instruction_capacity",
        lambda vector: math.nan if flat.part(vector, "law")[1] >= 0.0 else 0.0,
        flat,
        np.zeros((3, 3)),
        16.0,
    )
    assert row == {
        "key": "instruction_capacity",
        "level": 0.5,
        "method": "law",
        "value": None,
        "status": "no_slope",
    }


def test_a_root_past_the_ceiling_is_given_as_none() -> None:
    row = root_capacity_row(
        "instruction_capacity", _root, STACK, np.zeros((3, 3)), 16.0, ceiling=2.5
    )
    assert row["value"] is None
    assert row["status"] == "no_slope"


def test_the_measure_of_the_density_keeps_the_sign_of_the_change() -> None:
    # A falling law: the logit drops by 1.5 to full density, the odds of an error grow e^1.5 times.
    falling = odds_factor_row(
        lambda vector: 1.5 * float(STACK.part(vector, "law")[1]), STACK, np.zeros((3, 3))
    )
    assert falling["value"] == pytest.approx(math.exp(1.5))
    assert falling["change"] == pytest.approx(-1.5)
    # A rising law: the logit grows by 0.75, the odds of an error fall, the factor is below one.
    rising_stack = LawStack({"law": (np.array([3.0, 0.5]), np.eye(2))})
    rising = odds_factor_row(
        lambda vector: 1.5 * float(rising_stack.part(vector, "law")[1]),
        rising_stack,
        np.diag([0.0, 0.01, 0.0]),
    )
    half = 1.96 * 1.5 * 0.1
    assert rising["value"] == pytest.approx(math.exp(-0.75))
    assert rising["value"] < 1.0
    assert rising["interval"] == pytest.approx([math.exp(-0.75 - half), math.exp(-0.75 + half)])
    assert rising["status"] == "measured"


def test_a_measure_of_the_density_past_the_largest_float_is_none() -> None:
    # The logit drops by 800 to full density: e^800 does not hold in a float.
    steep = odds_factor_row(
        lambda vector: 800.0 * float(STACK.part(vector, "law")[1]), STACK, np.zeros((3, 3))
    )
    assert steep["value"] is None
    assert steep["interval"] == [None, None]
    assert steep["change"] == pytest.approx(-800.0)
