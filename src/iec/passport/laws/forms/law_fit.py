"""The fit of a form through the fitted bins of a block: least squares over their covariance.

The second stage of the passport draws a law through the coefficients of the bins of a block. The
fit is one for every form: the residuals ``Lᵀ·(β − f(x))`` are whitened by the root ``L`` of the
weight matrix ``P = L·Lᵀ``, the starts are tried in order and the smallest
``χ² = (β − f)ᵀ·P·(β − f)`` wins. The weighting gives ``P``: the pseudo-inverse of the full
covariance of the bins (generalised least squares, the method of the passport), the inverse own
variances of the points, or the identity. A parameter that the data push to a bound of its range is
held there: its value is the bound and it has no error. A form with such parameters is fitted by
``dogbox``, which stops on a bound the data push into, the others by ``trf``.

The uncertainties go by the delta method through the covariance of the bins: the estimates of a law
are functions of the bin coefficients, and their derivative by them is the sensitivity
``(JᵀPJ)⁻¹JᵀP`` over the free parameters, from the Gauss-Newton normal equations. This is the
procedure of the density law of P_sync, written once for every form.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from iec.passport.fitting.logit_fit import INFORMATION_RCOND
from iec.passport.laws.forms.law_form import LawForm

#: Relative distance to a bound at which a parameter is held on it.
HELD_TOLERANCE = 1e-6
#: The optimizer of a form with parameters that may be held: it stops on a bound.
HELD_METHOD = "dogbox"
#: The optimizer of a form without such parameters.
FREE_METHOD = "trf"


@dataclass(frozen=True, slots=True, eq=False)
class LawWeighting:
    """The weight matrix ``P`` of a fit and its root ``L``, ``P = L·Lᵀ``."""

    precision: np.ndarray
    root: np.ndarray


def covariance_weighting(block: np.ndarray) -> LawWeighting:
    """Return the weighting of generalised least squares: the pseudo-inverse of the covariance.

    The bin coefficients come from one fit and are correlated, so the weights come from the full
    block: ordinary least squares would understate the error. A direction of the block below
    ``INFORMATION_RCOND`` of its largest eigenvalue is not supported by the data and has no weight.

    Args:
        block: The covariance of the values of the points.
    """
    precision = np.linalg.pinv(block, rcond=INFORMATION_RCOND, hermitian=True)
    eigenvalues, vectors = np.linalg.eigh(precision)
    keep = eigenvalues > 0.0
    return LawWeighting(precision, vectors[:, keep] * np.sqrt(eigenvalues[keep]))


def variance_weighting(variances: np.ndarray) -> LawWeighting:
    """Return the weighting by the own variances of the points, without their mutual dependence.

    Args:
        variances: The variance of the value of every point, positive.
    """
    return LawWeighting(np.diag(1.0 / variances), np.diag(1.0 / np.sqrt(variances)))


def plain_weighting(size: int) -> LawWeighting:
    """Return the weighting of plain least squares: every point weighs one.

    Args:
        size: The number of points.
    """
    return LawWeighting(np.eye(size), np.eye(size))


@dataclass(frozen=True, slots=True, eq=False)
class LawFit:
    """The fit of a form through points.

    ``form`` - the name of the form; ``params`` - the estimates in the order of its parameters;
    ``chi2`` - the weighted sum of the squared residuals; ``held`` - the parameters held on a bound;
    ``sensitivity`` - the derivative of the estimates by the values of the points
    ``(parameters, points)``, zero rows for the held parameters; ``converged`` - the optimizer
    reached a solution and the normal equations are regular. Without convergence the sensitivity is
    zero and the estimates have no meaning.
    """

    form: str
    params: np.ndarray
    chi2: float
    held: tuple[int, ...]
    sensitivity: np.ndarray
    converged: bool


def fit_law(
    form: LawForm,
    x: np.ndarray,
    values: np.ndarray,
    weighting: LawWeighting,
    *,
    starts: Sequence[np.ndarray] | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> LawFit:
    """Fit a form through the points ``(x, values)``.

    Args:
        form: The form of the law.
        x: The abscissae of the points.
        values: The values of the points: the coefficients of the bins.
        weighting: The weight matrix of the points and its root.
        starts: The starts tried in order; ``None`` - those of the form.
        bounds: The lower and the upper bound of every parameter; ``None`` - those of the form.

    Returns:
        ``LawFit``; singular normal equations - a law that its points do not determine - give
        ``converged=False``.
    """
    lower, upper = form.bounds(x) if bounds is None else bounds
    candidates = form.starts(x, values) if starts is None else starts
    method = HELD_METHOD if form.held_candidates else FREE_METHOD
    size = len(form.parameters)
    sensitivity = np.zeros((size, len(values)))
    attempts = []
    for start in candidates:
        try:
            attempts.append(
                _least_squares(form, x, values, weighting.root, start, lower, upper, method)
            )
        except ValueError as error:
            # A form without a finite value at a point - the hyperbola at x = 0 - does not go
            # through the points from this start.
            if "not finite" not in str(error):
                raise
    if not attempts:
        return LawFit(form.name, np.full(size, np.nan), math.inf, (), sensitivity, False)
    result, chi2 = min(attempts, key=lambda pair: pair[1])
    params = np.asarray(result.x, dtype=float)
    if not result.success:
        return LawFit(form.name, params, chi2, (), sensitivity, False)
    held = tuple(held_parameters(result.x, lower, upper, form.held_candidates))
    free = [column for column in range(size) if column not in held]
    try:
        sensitivity[free] = _sensitivity(form.jacobian(result.x, x)[:, free], weighting.precision)
    except np.linalg.LinAlgError:
        return LawFit(form.name, params, chi2, held, np.zeros((size, len(values))), False)
    return LawFit(form.name, params, chi2, held, sensitivity, True)


def _least_squares(
    form: LawForm,
    x: np.ndarray,
    values: np.ndarray,
    root: np.ndarray,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    method: str,
) -> tuple[Any, float]:
    """Solve the least squares of one start on the whitened residuals; return ``χ²`` too.

    The residuals are ``Lᵀ(β − f(x))`` and their derivative ``−Lᵀ·J``; the start is clipped into the
    range.
    """
    # A form without a finite value at a point fails the check of the optimizer, not a warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = least_squares(
            lambda params: root.T @ (values - form.value(params, x)),
            np.clip(np.asarray(start, dtype=float), lower, upper),
            jac=lambda params: -root.T @ form.jacobian(params, x),
            bounds=(lower, upper),
            method=method,
        )
    return result, float(result.fun @ result.fun)


def held_parameters(
    params: np.ndarray, lower: np.ndarray, upper: np.ndarray, candidates: Sequence[int]
) -> list[int]:
    """Return the parameters among ``candidates`` whose value lies on a bound of its range.

    Such a parameter is not measured: the data push it to the edge of the range, so it is held
    there - its value is the bound, its sensitivity row is zero and it has no error. The fits of the
    sections that keep their own solver hold their parameters by the same rule.

    Args:
        params: The parameters of a law.
        lower: The lower bound of every parameter.
        upper: The upper bound of every parameter.
        candidates: The parameters that may be held.
    """
    held: list[int] = []
    for column in candidates:
        value = float(params[column])
        for bound in (float(lower[column]), float(upper[column])):
            if math.isfinite(bound) and abs(value - bound) <= HELD_TOLERANCE * max(1.0, abs(bound)):
                held.append(column)
                break
    return held


def _sensitivity(jacobian: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Return the sensitivity ``(JᵀPJ)⁻¹JᵀP`` of the estimates to the values of the points.

    Raises:
        np.linalg.LinAlgError: the normal equations are singular.
    """
    normal = jacobian.T @ precision @ jacobian
    return np.linalg.solve(normal, jacobian.T @ precision)
