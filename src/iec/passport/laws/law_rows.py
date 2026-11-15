"""The rows of the passport from the laws: covariance, half-widths, own errors, capacities.

The tools carry no formula of a model: every quantity keeps its laws in its own module. A half-width
is 1.96 standard errors by the delta method over the covariance of the coefficients and of the
variance of the request shift. The own error of a bin is its quasi-variance (Firth 2000): numbers
``q_k`` such that ``q_k + q_l`` gives the variance of the difference of bins ``k`` and ``l``. A
reference bin has an own error too: its zero comes from the coding alone. The module gives also the
facts of a fit with the spread of the request shift by the load and the row of a capacity by a law;
the approximations of the spread live in ``spread_laws``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from iec.passport.fitting.chain.fit_notes import SPREAD_MERGED
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.gls_tools import STATUS_EXTRAPOLATED, STATUS_MEASURED

#: Half-width of the 95% interval in standard errors.
HALF = 1.96
#: The quality level of the capacities. At it the share of successes over all requests and the success
#: of the median request reach one half at the same logit, whatever the variance of the request shift
#: and whatever its symmetric distribution, so a capacity does not depend on that variance.
LEVEL = 0.5
#: The logit of the median request at which a capacity of the level :data:`LEVEL` is read.
LEVEL_THRESHOLD = 0.0


def joint_covariance(fit: VariableFit) -> np.ndarray:
    """Return the covariance over ``(theta, sigma2)``; without the shift its row is zero.

    Args:
        fit: The fit of a quantity.
    """
    size = len(fit.names)
    joint = np.zeros((size + 1, size + 1))
    matrix = np.asarray(fit.covariance, dtype=float)
    joint[:size, :size] = (matrix + matrix.T) / 2.0
    if fit.sigma2_covariance:
        row = np.asarray(fit.sigma2_covariance, dtype=float)
        joint[size, :] = row
        joint[:, size] = row
    return joint


def half_width(gradient: np.ndarray, joint: np.ndarray) -> float:
    """Return 1.96 standard errors of a quantity; a shorter gradient has zeros at its end.

    Args:
        gradient: The derivative of the quantity by the coefficients and the variance of the shift.
        joint: The joint covariance (:func:`joint_covariance`).
    """
    full = np.zeros(len(joint))
    full[: len(gradient)] = gradient
    return HALF * math.sqrt(max(float(full @ joint @ full), 0.0))


def spread_summary(fit: VariableFit) -> dict[str, Any]:
    """Return the spread by the load for the passport: the variance by the groups of every axis.

    The variance of a group is taken at this group with the other two axes averaged over the requests.

    Args:
        fit: A fit of the method ``glmm_heterogeneous``: it has ``spread``.

    Returns:
        ``parameters`` and ``loglik_gain`` of the spread, and the rows of the groups of every axis
        ``V``, ``P``, ``M``: ``bins``, ``center``, ``share``, ``failing``, ``sigma2``, ``half``.
    """
    spread = fit.spread
    assert spread is not None
    size = len(fit.names)
    block = np.asarray(fit.spread_covariance, dtype=float)[size:, size:]
    gamma = np.asarray(spread.gamma, dtype=float)
    result: dict[str, Any] = {
        "parameters": len(gamma) - 1,
        "loglik_gain": 2.0 * (float(fit.loglik or 0.0) - float(fit.loglik_common or 0.0)),
    }
    for position, (letter, axis) in enumerate(zip("VPM", spread.axes, strict=True)):
        rows: list[dict[str, Any]] = []
        for slot, center in enumerate(axis.centers):
            values: list[float | None] = [None, None, None]
            values[position] = center
            row = spread.row_at((values[0], values[1], values[2]))
            sigma2 = math.exp(2.0 * float(row @ gamma))
            rows.append(
                {
                    "bins": list(axis.groups[slot]),
                    "center": center,
                    "share": axis.shares[slot],
                    "failing": axis.failing[slot],
                    "sigma2": sigma2,
                    "half": HALF * 2.0 * sigma2 * math.sqrt(max(float(row @ block @ row), 0.0)),
                }
            )
        result[letter] = rows
    return result


def own_variances(block: np.ndarray, implicit_reference: bool) -> np.ndarray:
    """Return the quasi-variances of the bins of one axis.

    Args:
        block: The covariance of the coefficients of the axis.
        implicit_reference: The axis has a reference bin without a column: its quasi-variance
            comes first in the result.

    Returns:
        The quasi-variances, the reference bin first if there is one.
    """
    columns = len(block)
    size = columns + 1 if implicit_reference else columns
    if size < 2:
        return np.zeros(size)
    pairs: list[tuple[int, int, float]] = []
    for k in range(size):
        for other in range(k + 1, size):
            row = np.zeros(columns)
            if implicit_reference:
                if k > 0:
                    row[k - 1] = -1.0
                row[other - 1] = 1.0
            else:
                row[k] = -1.0
                row[other] = 1.0
            pairs.append((k, other, max(float(row @ block @ row), 1e-300)))

    def residual(log_q: np.ndarray) -> np.ndarray:
        """Return the misfits ``ln(q_k + q_l) − ln var(k − l)`` of all pairs of bins."""
        q = np.exp(log_q)
        return np.array([math.log(q[k] + q[other]) - math.log(value) for k, other, value in pairs])

    start = np.log(np.full(size, max(float(np.mean(np.diag(block))), 1e-12) / 2.0))
    return np.exp(least_squares(residual, start).x)


def fit_summary(fit: VariableFit) -> dict[str, Any]:
    """Return the facts of a fit for the passport: method, variance of the shift, sizes.

    Args:
        fit: The fit of a quantity.

    Returns:
        ``method``, ``sigma2`` and its half-width, the sizes of the fit, the dropped columns, the
        log-likelihood and, for a fit with the spread by the load, ``spread``; its ``merged`` lists
        the groups merged because the fit stopped on the floor of the spread (decision 4 of the
        passport standard).
    """
    joint = joint_covariance(fit)
    summary: dict[str, Any] = {
        "method": fit.method,
        "sigma2": fit.sigma2,
        "sigma2_half": HALF * math.sqrt(max(float(joint[-1, -1]), 0.0)),
        "steps": fit.steps,
        "rows": fit.rows,
        "points": fit.points,
        "failures": fit.failures,
        "trials": fit.trials,
        "parameters": len(fit.names),
        "dropped": list(fit.dropped),
        "loglik": fit.loglik,
    }
    if fit.spread is not None:
        summary["spread"] = spread_summary(fit)
        merged = [note.message for note in fit.notes if note.code == SPREAD_MERGED]
        if merged:
            summary["spread"]["merged"] = merged
    return summary


def capacity_row(
    key: str,
    level: float,
    log_value: float,
    half: float,
    support: float,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Return a capacity by a law from its binary logarithm.

    Args:
        key: The name of the capacity.
        level: The quality level ``q``.
        log_value: ``log₂`` of the capacity divided by ``scale``.
        half: The half-width of ``log_value``.
        support: The edge of the measured region in the units of the capacity.
        scale: The unit of the logarithm: 20 words for a length counted from 20 words.

    Returns:
        ``value``, the interval on the logarithmic scale, ``plus_minus`` - the half-width in the
        units of the capacity, ``status`` - ``measured`` inside the measured region, otherwise
        ``extrapolated`` (the article, section 4.5).
    """
    value = scale * 2.0**log_value
    return {
        "key": key,
        "level": level,
        "method": "law",
        "value": value,
        "interval": [scale * 2.0 ** (log_value - half), scale * 2.0 ** (log_value + half)],
        "plus_minus": math.log(2.0) * value * half,
        "status": STATUS_MEASURED if value <= support else STATUS_EXTRAPOLATED,
    }
