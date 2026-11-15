"""Generalised least squares through the fitted bins: the points, the solver, the sensitivity.

The second stage of the passport is **not** a Bernoulli fit but a measuring treatment of the
already estimated bins: a law is drawn through the bin coefficients of a block by generalised least
squares over the covariance of the coefficients. The value of a law at the zero of its axis is the
shift of the block; moving it into the level is a calibration, not a change of the model: the
prediction of every trial stays the same.

Uncertainties are by the delta method through the first-stage covariance: every second-stage
estimate is a function of the bin coefficients, and its sensitivity to them is known - explicitly
for a linear fit, through the Gauss-Newton normal equations for a non-linear one. The module holds
also the statuses of a published quantity.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import least_squares

#: Statuses of published quantities. ``measured`` - the quantity is computed; the others say
#: honestly why there is no number or what limits it.
STATUS_MEASURED = "measured"
STATUS_INSUFFICIENT = "insufficient_support"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_EXTRAPOLATED = "extrapolated"


def gls_sensitivity(
    jacobian: np.ndarray, precision: np.ndarray, idx: Sequence[int], width: int
) -> np.ndarray:
    """Return the sensitivity of the law estimates to the raw coefficients: ``(JᵀPJ)⁻¹JᵀP``.

    Second-stage estimates are functions of the bin coefficients, and at the solution their
    derivative with respect to these coefficients equals this matrix (the Gauss-Newton normal
    equations). Through it the delta method carries the first-stage covariance to every passport
    quantity: the second stage has no separate source of uncertainty.

    Args:
        jacobian: The derivatives of the law at its points by its free parameters.
        precision: The weight matrix of the points.
        idx: The columns of the points among all coefficients of the fit.
        width: The number of all coefficients.

    Returns:
        The rows of the free parameters over all coefficients; zero outside ``idx``.

    Raises:
        np.linalg.LinAlgError: the normal equations are singular.
    """
    normal = jacobian.T @ precision @ jacobian
    rows = np.linalg.solve(normal, jacobian.T @ precision)
    full = np.zeros((rows.shape[0], width), dtype=float)
    full[:, list(idx)] = rows
    return full


def named_points(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    prefix: str,
    abscissa: Callable[[int], float],
    abscissae: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    """Return the points of a series by name: abscissae, values, Σ block, indices, names.

    **Names** are walked, not numbers: the reference density bin is not the first one, and its
    name is absent from the fit. A bin without trials (zero variance - an empty design column)
    is not a point: its coefficient is not measured. A bin with separation has a huge error,
    and generalised least squares gives it a negligible weight by itself - no separate rule is
    needed.

    Args:
        theta: The coefficients by name.
        covariance: Their covariance in the order of ``names``.
        names: The columns: the bins of the series and any others.
        prefix: The prefix of the names of the series, such as ``p_bin_``.
        abscissa: The abscissa of a bin by its number, for a name without an entry in
            ``abscissae``.
        abscissae: The abscissae from outside by name. Sorting by abscissa is after the
            substitution, otherwise the points of the law and the rows of the sensitivity would
            part by index.

    Returns:
        The abscissae, the values, the covariance block of the points, their indices in ``names``
        and their names, in the order of the abscissae.
    """
    order = list(names)

    def place(name: str) -> float:
        """Return the abscissa of a bin: from outside if given, otherwise by its number."""
        if abscissae is not None and name in abscissae:
            return float(abscissae[name])
        return float(abscissa(int(name[len(prefix) :])))

    selected = sorted(
        (place(name), name)
        for name in order
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
    )
    selected = [
        (x, name) for x, name in selected if covariance[order.index(name), order.index(name)] > 0.0
    ]
    x = np.array([value for value, _ in selected], dtype=float)
    beta = np.array([float(theta[name]) for _, name in selected], dtype=float)
    idx = [order.index(name) for _, name in selected]
    block = covariance[np.ix_(idx, idx)] if idx else np.zeros((0, 0))
    return x, beta, block, idx, [name for _, name in selected]


def gls_least_squares(
    curve: Callable[[np.ndarray, np.ndarray], np.ndarray],
    jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    root: np.ndarray,
    method: str = "trf",
) -> tuple[Any, float]:
    """Solve non-linear GLS on the whitened residuals ``Lᵀ(β − f(x))``; return ``χ²`` too.

    Args:
        curve: The law ``f(params, x)``.
        jacobian: Its derivatives by the parameters ``(points, parameters)``.
        start: The start; it is clipped into the range.
        lower: The lower bound of every parameter.
        upper: The upper bound of every parameter.
        x: The abscissae of the points.
        beta: The values of the points.
        root: The root ``L`` of the weight matrix ``P = L·Lᵀ``.
        method: ``"trf"`` by default; ``"dogbox"`` for a law whose parameter may be held on a
            bound of its range: it treats a bound the data push into as active and stops on it,
            while ``"trf"`` keeps the parameters strictly inside and approaches the bound only
            asymptotically.

    Returns:
        The result of ``scipy.optimize.least_squares`` and ``χ²`` at the solution.
    """
    result = least_squares(
        lambda params: root.T @ (beta - curve(params, x)),
        np.clip(np.asarray(start, dtype=float), lower, upper),
        jac=lambda params: -root.T @ jacobian(params, x),
        bounds=(lower, upper),
        method=method,
    )
    return result, float(result.fun @ result.fun)


def delta_se(gradient: np.ndarray, covariance: np.ndarray) -> float:
    """Return the SE by the delta method: ``√(∇′·Cov·∇)``; a noisy negative square is zero.

    Args:
        gradient: The derivative of the quantity by the coefficients.
        covariance: The covariance of the coefficients.
    """
    return math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))
