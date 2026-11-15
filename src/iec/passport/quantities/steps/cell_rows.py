"""The rows of the cells of the length × word-number grid: raw, minus the shift, the law, the counts.

A row of a cell publishes its coefficient with its own half-width, the same minus the shift of the
grid with the half-width of the difference, the law of the grid at the cell and the counts of the
cell; a cell without a coefficient is ``excluded``. P_sync and H publish their grids so; the law of
the grid is the formula of the model of each, and the section gives it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.law_rows import HALF, half_width
from iec.passport.trials.design.folded_design import GridCell, VariableDesign


def cell_rows(
    design: VariableDesign,
    fit: VariableFit,
    shift: float,
    shift_row: np.ndarray,
    joint: np.ndarray,
    law_at: Callable[[GridCell, Mapping[str, float]], float],
) -> list[dict[str, Any]]:
    """Return the rows of the cells of the grid.

    Args:
        design: The folded matrix: the cells of the grid and their centres.
        fit: The fit of the quantity.
        shift: The shift of the grid, subtracted from the coefficients.
        shift_row: The sensitivity of the shift to the coefficients.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        law_at: The law of the grid at a cell, from the cell and its centre ``x``, ``u``.

    Returns:
        The rows in the order of the cells of the design.
    """
    names = list(fit.names)
    rows: list[dict[str, Any]] = []
    for cell in design.cells:
        point = design.abscissae[cell.name]
        row: dict[str, Any] = {
            "name": cell.name,
            "v_bin": cell.v_bin,
            "s_bin": cell.s_bin,
            "words": [cell.lower_words, cell.upper_words],
            "x": point["x"],
            "u": point["u"],
            "trials": cell.trials,
            "failures": cell.failures,
            "requests": cell.points_with_failures,
        }
        estimate = fit.coefficient(cell.name)
        if estimate is None:
            rows.append({**row, "status": "excluded"})
            continue
        gradient = -shift_row.copy()
        gradient[names.index(cell.name)] += 1.0
        rows.append(
            {
                **row,
                "status": "law",
                "raw": estimate[0],
                "raw_half": HALF * estimate[1],
                "restored": estimate[0] - shift,
                "restored_half": half_width(gradient, joint),
                "law": law_at(cell, point),
            }
        )
    return rows
