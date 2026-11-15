"""The rows of the bins of an axis: the reference, the law, an outlier, an excluded bin."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.fitting.chain.fit_paths import METHOD_GLMM
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.quantities.steps.axis_rows import axis_rows
from iec.passport.trials.design.folded_design import AxisBin, AxisTable, VariableDesign


def _design_and_fit() -> tuple[VariableDesign, VariableFit]:
    """Four bins of M: the reference, two measured and one without a coefficient."""
    bins = tuple(
        AxisBin(number, m, m, f"m_bin_{number}", 100, 10, 20)
        for number, m in ((1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0))
    )
    design = VariableDesign(
        variable="P_sync",
        x=np.zeros((1, 2)),
        failures=np.zeros(1),
        trials=np.ones(1),
        run=np.zeros(1, dtype=np.int64),
        point_keys=("a",),
        names=("m_bin_2", "m_bin_3"),
        cells=(),
        axes={"M": AxisTable("M", bins, 1)},
        abscissae={item.name: {"m": item.upper} for item in bins},
    )
    fit = VariableFit(
        variable="P_sync",
        method=METHOD_GLMM,
        names=("m_bin_2", "m_bin_3"),
        theta=(-1.0, -2.0),
        se=(0.1, 0.1),
        covariance=((0.01, 0.0), (0.0, 0.01)),
        dropped=(),
        sigma2=1.0,
        sigma2_covariance=(0.0, 0.0, 0.01),
        steps=1,
        rows=1,
        points=1,
        failures=10,
        trials=100,
    )
    return design, fit


def test_the_rows_of_an_axis_keep_the_reference_and_mark_their_bins() -> None:
    design, fit = _design_and_fit()
    joint = np.diag([0.01, 0.01, 0.01])
    rows = axis_rows(
        design, fit, "M", "m", ["m_bin_2"], lambda m: -float(np.log2(m)), 0.5, np.zeros(2), joint
    )
    statuses = [row["status"] for row in rows]
    assert statuses == ["reference", "law", "outlier", "excluded"]
    assert rows[0]["raw"] == 0.0
    assert rows[0]["own_half"] > 0.0
    assert rows[1]["restored"] == pytest.approx(-1.5)
    assert rows[1]["law"] == pytest.approx(-1.0)
    assert rows[2]["restored_half"] == pytest.approx(1.96 * 0.1)
    assert "raw" not in rows[3]


def test_the_error_of_a_restored_bin_carries_the_error_of_the_shift() -> None:
    design, fit = _design_and_fit()
    joint = np.diag([0.01, 0.01, 0.01])
    shift_row = np.array([0.0, 1.0])
    rows = axis_rows(design, fit, "M", "m", ["m_bin_2"], lambda m: 0.0, 0.0, shift_row, joint)
    # The reference bin carries the error of the shift alone; m_bin_2 the sum of both.
    assert rows[0]["restored_half"] == pytest.approx(1.96 * 0.1)
    assert rows[1]["restored_half"] == pytest.approx(1.96 * np.sqrt(0.02))
    assert rows[2]["restored_half"] == pytest.approx(0.0, abs=1e-12)
