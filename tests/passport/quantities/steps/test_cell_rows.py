"""The rows of the cells of the grid: a measured cell with the law, a cell without a coefficient."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.fitting.chain.fit_paths import METHOD_GLMM
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.quantities.steps.cell_rows import cell_rows
from iec.passport.trials.design.folded_design import GridCell, VariableDesign


def test_a_cell_gives_its_coefficient_minus_the_shift_and_the_law_of_its_slice() -> None:
    cells = (
        GridCell(3, 1, "vs_3_1", 41, 80, 100, 10, 20),
        GridCell(3, 2, "vs_3_2", 41, 80, 100, 10, 20),
    )
    design = VariableDesign(
        variable="H",
        x=np.zeros((1, 1)),
        failures=np.zeros(1),
        trials=np.ones(1),
        run=np.zeros(1, dtype=np.int64),
        point_keys=("a",),
        names=("vs_3_1",),
        cells=cells,
        axes={},
        abscissae={"vs_3_1": {"x": 4.0, "u": 2.0}, "vs_3_2": {"x": 8.0, "u": 2.0}},
    )
    fit = VariableFit(
        variable="H",
        method=METHOD_GLMM,
        names=("vs_3_1",),
        theta=(5.0,),
        se=(0.5,),
        covariance=((0.25,),),
        dropped=(),
        sigma2=1.0,
        sigma2_covariance=(0.0, 0.01),
        steps=1,
        rows=1,
        points=1,
        failures=10,
        trials=100,
    )
    joint = np.diag([0.25, 0.01])
    rows = cell_rows(
        design, fit, 2.0, np.zeros(1), joint, lambda cell, point: float(np.log2(point["x"]))
    )
    assert [row["status"] for row in rows] == ["law", "excluded"]
    assert rows[0]["raw"] == 5.0
    assert rows[0]["raw_half"] == pytest.approx(1.96 * 0.5)
    assert rows[0]["restored"] == pytest.approx(3.0)
    assert rows[0]["restored_half"] == pytest.approx(1.96 * 0.5)
    assert rows[0]["law"] == pytest.approx(2.0)
    assert rows[1]["words"] == [41, 80]
    assert "raw" not in rows[1]
