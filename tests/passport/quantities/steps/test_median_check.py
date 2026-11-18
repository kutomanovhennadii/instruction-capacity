"""The check of the variance by the load against the median request of the data."""

from __future__ import annotations

from iec.passport.quantities.steps.median_check import (
    AxisBins,
    Request,
    median_comparison,
    spread_against_data,
)


def _requests(worse_share: float) -> list[Request]:
    """Return requests of density bins 1 (reference) and 2 in two cells of the other axes."""
    found = []
    for cell in (1, 2):
        for _ in range(20):
            found.append(Request((cell, 1, cell), 100.0, 10.0))
            found.append(Request((cell, 2, cell), 100.0, 100.0 * worse_share))
    return found


def test_the_median_request_of_a_worse_bin_lies_below_the_reference() -> None:
    comparison = median_comparison(_requests(0.3), 1, 1)
    assert comparison[2] < 0.0


def test_a_bin_against_the_median_request_is_named_and_one_along_it_is_not() -> None:
    axes = {"P": AxisBins(1, {1: "p_bin_1", 2: "p_bin_2"})}
    assert spread_against_data(_requests(0.3), axes, {"p_bin_2": 0.5})[0].startswith("p_bin_2")
    assert spread_against_data(_requests(0.3), axes, {"p_bin_2": -0.5}) == []


def test_requests_with_few_trials_do_not_enter_the_check() -> None:
    few = [Request((1, number, 1), 10.0, 5.0) for number in (1, 2) for _ in range(5)]
    assert spread_against_data(few, {"P": AxisBins(1, {2: "p_bin_2"})}, {"p_bin_2": 1.0}) == []
