"""Matrices of the three quantities in the passport layout: names, grid, references, centres."""

from __future__ import annotations

import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin
from iec.passport.trials.place_trials import PointTrials
from iec.passport.trials.design.folded_design import build_design

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 640),
    density_bounds=doubling_bounds(1, 16),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 16)),
)
# Point a: v = 160, p = 1/4 - the reference density bin of the passport (bin 3), m = 10 - bin 4.
# Point b: v = 160, p = 1 - bin 5, m = 4 - bin 3; it has no background words.
POINTS = [
    PointTrials(
        key="a",
        D=40,
        n=4,
        m=10,
        sync=((1, 10, 1, 21.8), (3, 20, 4, 97.0)),
        rule=((1, 1, 1, 0, 0.0), (3, 2, 2, 1, 9.6), (3, 9, 3, 2, 14.4)),
        background=((1, 9, 1, 19.6), (3, 15, 0, 72.0)),
        totals={},
    ),
    PointTrials(
        key="b",
        D=4,
        n=40,
        m=4,
        sync=((2, 10, 0, 39.3),),
        rule=((2, 4, 10, 3, 39.3),),
        background=(),
        totals={},
    ),
]
WEIGHTS = {"a": 1.0, "b": 1.0}


def test_sync_design_has_passport_names_and_keeps_every_place() -> None:
    design = build_design(POINTS, "P_sync", LAYOUT, WEIGHTS)

    assert design.names == ("vs_4_1", "vs_4_2", "vs_4_3", "p_bin_5", "m_bin_3", "m_bin_4")
    assert (design.trials.sum(), design.failures.sum()) == (40.0, 5.0)
    assert [(cell.name, cell.trials, cell.failures) for cell in design.cells] == [
        ("vs_4_1", 10, 1),
        ("vs_4_2", 10, 0),
        ("vs_4_3", 20, 4),
    ]
    assert (design.axes["P"].reference, design.axes["M"].reference) == (3, 1)


def test_rule_design_pools_rule_numbers_by_doubling() -> None:
    design = build_design(POINTS, "H", LAYOUT, WEIGHTS)

    assert len(design.trials) == 4
    table = design.axes["R"]
    assert [(item.number, item.trials, item.failures) for item in table.bins] == [
        (1, 1, 0),
        (2, 2, 1),
        (3, 10, 3),
        (4, 3, 2),
    ]
    assert table.reference == 1
    assert design.names[-3:] == ("r_bin_2", "r_bin_3", "r_bin_4")


def test_point_without_background_gives_no_f_rows() -> None:
    design = build_design(POINTS, "F", LAYOUT, WEIGHTS)

    assert design.point_keys == ("a",)
    assert design.trials.sum() == 24.0
    assert "R" not in design.axes


def test_abscissae_follow_the_passport_format() -> None:
    design = build_design(POINTS, "P_sync", LAYOUT, WEIGHTS)

    assert design.abscissae["vs_4_1"] == pytest.approx({"x": 2.0**2.18, "u": 3.0})
    assert design.abscissae["p_bin_3"] == pytest.approx({"p": 0.25})
    assert design.abscissae["m_bin_4"] == pytest.approx({"m": 10.0})


def test_missing_weight_is_an_error() -> None:
    with pytest.raises(ContractViolationError) as caught:
        build_design(POINTS, "P_sync", LAYOUT, {"a": 1.0})

    assert caught.value.error_code == "POINT_WEIGHT_MISSING"


def test_unknown_variable_is_an_error() -> None:
    with pytest.raises(ContractViolationError) as caught:
        build_design(POINTS, "G", LAYOUT, WEIGHTS)

    assert caught.value.error_code == "VARIABLE_UNKNOWN"
