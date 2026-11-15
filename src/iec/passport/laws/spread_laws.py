"""The approximations of the variance of the request shift by the groups of the axes of the load.

The passport gives the variance of the shift of a request by the groups of every axis ``V``, ``P``,
``M`` (``law_rows.spread_summary``); an approximation draws a smooth curve through these groups for
the eye. The forms come from the assignment of forms (``config/passport_laws.yaml``): the logarithmic
law of the collection is published as the line ``a + b·log₂ x``, the fractional hyperbola as
``c + A/(1 + x/x₀)``; a hyperbola whose ``x₀`` falls to the lower bound of its search degenerates into
the inverse law ``c + A/x``. The fit of an approximation goes by the residuals in ``ln σ²``, every
group weighed by its relative precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.law_assignment import LAW_FORM_UNSUPPORTED, LawAssignment
from iec.passport.laws.law_rows import HALF

#: The forms of the approximation of the variance by the groups of an axis, as the passport
#: publishes them: a line in ``log₂`` of the axis, ``a + b·log₂ x``, the fractional hyperbola
#: ``c + A/(1 + x/x₀)`` and the inverse law ``c + A/x`` of a degenerate hyperbola.
SPREAD_LINE = "line"
SPREAD_HYPERBOLA = "hyperbola"
SPREAD_INVERSE = "inverse"
#: The axes of the spread in the order of the passport.
SPREAD_AXES = ("V", "P", "M")
#: The published name of an approximation by the name of its form in the collection.
SPREAD_FORM_NAMES = {"logarithmic": SPREAD_LINE, "fractional_hyperbolic": SPREAD_HYPERBOLA}
#: The parameters of a form in the order of the fit.
_SPREAD_PARAMETERS = {
    SPREAD_LINE: ("a", "b"),
    SPREAD_HYPERBOLA: ("c", "A", "x0"),
    SPREAD_INVERSE: ("c", "A"),
}
#: The lower bound of ``x₀`` of a hyperbola, as a share of the smallest centre of the groups.
_SPREAD_X0_FLOOR = 1e-3
#: The residual of a group at which a law is not positive: far beyond any honest residual.
_SPREAD_OUTSIDE = 1e6


def spread_forms(assignment: LawAssignment, quantity: str) -> dict[str, str]:
    """Return the published forms of the approximations of the variance of a quantity by axis.

    Args:
        assignment: The forms of the laws of the passport.
        quantity: The quantity: ``P_sync``, ``H`` or ``F``.

    Returns:
        The form of every axis ``V``, ``P``, ``M``: :data:`SPREAD_LINE` or :data:`SPREAD_HYPERBOLA`.

    Raises:
        ConfigError: ``LAW_FORM_UNSUPPORTED`` - the assignment gives an axis a form that has no
            approximation of the variance.
        ContractViolationError: ``LAW_BLOCK_UNKNOWN`` - the assignment names no form of an axis.
    """
    forms: dict[str, str] = {}
    for axis in SPREAD_AXES:
        name = assignment.spread_form_of(quantity, axis).name
        published = SPREAD_FORM_NAMES.get(name)
        if published is None:
            known = ", ".join(sorted(SPREAD_FORM_NAMES))
            raise ConfigError(
                f"the spread of {quantity} by the axis {axis} is approximated by {known}, "
                f"not by '{name}'",
                error_code=LAW_FORM_UNSUPPORTED,
            )
        forms[axis] = published
    return forms


def _spread_law(form: str, params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return the values of an approximation of the variance at the loads ``x``.

    Args:
        form: The published form: line, hyperbola or inverse law.
        params: Its parameters in the order of :data:`_SPREAD_PARAMETERS`.
        x: The loads.
    """
    if form == SPREAD_LINE:
        line: np.ndarray = params[0] + params[1] * np.log2(x)
        return line
    if form == SPREAD_INVERSE:
        inverse: np.ndarray = params[0] + params[1] / x
        return inverse
    hyperbola: np.ndarray = params[0] + params[1] / (1.0 + x / params[2])
    return hyperbola


def _fit_spread_law(form: str, x: np.ndarray, y: np.ndarray, scale: np.ndarray) -> Any:
    """Fit one approximation by the residuals in ``ln σ²``; return the result of the fit.

    Args:
        form: The published form.
        x: The centres of the groups.
        y: The variances of the groups.
        scale: The relative precision of every group, the denominator of its residual.
    """

    def residuals(params: np.ndarray) -> np.ndarray:
        """Return ``(ln σ² − ln law)/s``; a law that is not positive gets a huge residual."""
        value = _spread_law(form, params, x)
        if np.any(value <= 0.0):
            return np.full(len(x), _SPREAD_OUTSIDE)
        result: np.ndarray = (np.log(y) - np.log(value)) / scale
        return result

    if form == SPREAD_LINE:
        start = np.array([float(np.median(y)), 0.0])
        bounds = (np.full(2, -np.inf), np.full(2, np.inf))
    elif form == SPREAD_INVERSE:
        start = np.array([0.5 * float(y.min()), float(y.max()) * float(x.min())])
        bounds = (np.zeros(2), np.full(2, np.inf))
    else:
        start = np.array([0.5 * float(y.min()), float(y.max()), float(x.min())])
        bounds = (
            np.array([0.0, 0.0, _SPREAD_X0_FLOOR * float(x.min())]),
            np.array([np.inf, np.inf, 1e3 * float(x.max())]),
        )
    return least_squares(
        residuals,
        start,
        bounds=bounds,
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=20000,
    )


def spread_laws(spread: Mapping[str, Any], forms: Mapping[str, str]) -> dict[str, Any]:
    """Return the approximations of the variance by the groups of the axes, for the figures.

    The residual of a group is ``(ln σ² − ln law)/s`` with ``s = half/(1.96·σ²)``: the passport gives
    the interval of ``ln σ``, so every group weighs by its relative precision. The approximations serve
    the eye: the threshold of a capacity reads the variance of the groups from the fit. An axis with no
    more groups than the parameters of its form gets no approximation. A hyperbola whose ``x₀`` stops
    at the lower bound of its search has degenerated: then the inverse law ``c + A/x`` is fitted anew
    and given in its place, because ``A`` and ``x₀`` of such a hyperbola mean nothing apart.

    Args:
        spread: The summary of the spread, ``law_rows.spread_summary``.
        forms: The form of every axis ``V``, ``P``, ``M``: :data:`SPREAD_LINE` or
            :data:`SPREAD_HYPERBOLA` (:func:`spread_forms`).

    Returns:
        By axis: ``form``, ``params``, ``chi2``, ``n_points``.
    """
    result: dict[str, Any] = {}
    for letter, form in forms.items():
        rows = spread[letter]
        names = _SPREAD_PARAMETERS[form]
        if len(rows) <= len(names):
            continue
        x = np.array([float(row["center"]) for row in rows])
        y = np.array([float(row["sigma2"]) for row in rows])
        half = np.array([float(row["half"]) for row in rows])
        scale = np.maximum(half / (HALF * y), 1e-12)
        found = _fit_spread_law(form, x, y, scale)
        floor = _SPREAD_X0_FLOOR * float(x.min())
        if form == SPREAD_HYPERBOLA and float(found.x[2]) <= 1.01 * floor:
            form, names = SPREAD_INVERSE, _SPREAD_PARAMETERS[SPREAD_INVERSE]
            found = _fit_spread_law(form, x, y, scale)
        result[letter] = {
            "form": form,
            "params": {name: float(value) for name, value in zip(names, found.x, strict=True)},
            "chi2": float(found.fun @ found.fun),
            "n_points": len(rows),
        }
    return result
