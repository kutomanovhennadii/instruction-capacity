"""The rows of the bins of an axis in the passport: raw, minus the shift, own errors, the law.

A row of a bin publishes the coefficient of the fit, the same minus the shift of the block, the
half-width of each, the law at the centre of the bin and the counts of the bin. The own error of a
bin is its quasi-variance, and the reference bin has one too: its zero comes from the coding alone.
A bin of the law has the status ``law``, a measured bin clipped from the law as an outlier
``outlier``, the reference bin ``reference``, a bin without a coefficient ``excluded``. P_sync and H
publish their axes so; F has its own design of the bins and its own rows (``background_section``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection
from typing import Any

import numpy as np

from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.law_rows import HALF, half_width, own_variances
from iec.passport.trials.design.folded_design import VariableDesign


def axis_rows(
    design: VariableDesign,
    fit: VariableFit,
    letter: str,
    key: str,
    in_law: Collection[str],
    law_at: Callable[[float], float],
    shift: float,
    shift_row: np.ndarray,
    joint: np.ndarray,
) -> list[dict[str, Any]]:
    """Return the rows of the bins of an axis.

    Args:
        design: The folded matrix: the table of the bins of the axis and their centres.
        fit: The fit of the quantity.
        letter: The axis: ``P``, ``M`` or ``R``.
        key: The name of the centre of a bin in the abscissae: ``p``, ``m`` or ``r``.
        in_law: The names of the bins the law goes through.
        law_at: The law of the axis at a centre.
        shift: The shift of the block, subtracted from the coefficients.
        shift_row: The sensitivity of the shift to the coefficients.
        joint: The joint covariance of the coefficients and of the variance of the shift.

    Returns:
        The rows in the order of the bins of the axis.
    """
    names = list(fit.names)
    table = design.axes[letter]
    present = [item.name for item in table.bins if item.name in names]
    index = [names.index(name) for name in present]
    own = own_variances(joint[np.ix_(index, index)], implicit_reference=True)
    rows: list[dict[str, Any]] = []
    for item in table.bins:
        center = design.abscissae[item.name][key]
        row: dict[str, Any] = {
            "name": item.name,
            "bin": item.number,
            "center": center,
            "trials": item.trials,
            "failures": item.failures,
            "requests": item.points_with_failures,
        }
        gradient = -shift_row.copy()
        if item.number == table.reference:
            raw, status, q = 0.0, "reference", own[0]
        elif item.name in names:
            gradient[names.index(item.name)] += 1.0
            raw = fit.coefficient(item.name)[0]  # type: ignore[index]
            status = "law" if item.name in in_law else "outlier"
            q = own[present.index(item.name) + 1]
        else:
            rows.append({**row, "status": "excluded"})
            continue
        rows.append(
            {
                **row,
                "status": status,
                "raw": raw,
                "own_half": HALF * math.sqrt(q),
                "restored": raw - shift,
                "restored_half": half_width(gradient, joint),
                "law": float(law_at(center)),
            }
        )
    return rows
