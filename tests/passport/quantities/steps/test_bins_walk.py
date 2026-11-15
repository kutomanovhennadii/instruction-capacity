"""The capacity by the bins of the number of rules: inside the bins, past them, below one rule."""

from __future__ import annotations

import pytest

from iec.passport.quantities.steps.bins_walk import capacity_by_bin_walk

#: The reference bin and two bins: the price grows to one and to three logits.
ROWS = [(1.0, 0.0), (2.0, -1.0), (4.0, -3.0)]


def test_the_reserve_spent_inside_a_bin_gives_its_share() -> None:
    found = capacity_by_bin_walk(ROWS, 0.0, 2.0)
    assert found["status"] == "measured"
    assert found["value"] == pytest.approx(3.0)


def test_a_reserve_the_bins_do_not_spend_gives_a_lower_bound() -> None:
    found = capacity_by_bin_walk(ROWS, 0.0, 5.0)
    assert found == {
        "value": None,
        "status": "beyond_support",
        "lower_bound": 4,
        "reserve_left": 2.0,
    }


def test_no_reserve_is_below_one_rule() -> None:
    assert capacity_by_bin_walk(ROWS, 0.0, 0.0) == {"value": None, "status": "below_one_rule"}
