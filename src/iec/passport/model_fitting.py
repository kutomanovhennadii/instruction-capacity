"""The result of the fit of a passport model and the fit of the public model ``vspmr_glmm_v1``.

The fit: the plain logit as the start, then the GLMM fit with the shift of a point inside the
likelihood (:mod:`iec.passport.glmm_fit`). The variance of the shift and its SE come from the
fit; the covariance is the cluster sandwich by points around the Hessian of the marginal
likelihood. There is no fallback: when a fit does not converge, the passport is not computed,
and the error says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.infrastructure.logging import LogContext, get_logger
from iec.passport.design import Design
from iec.passport.glmm_fit import fit_glmm
from iec.passport.logit_fit import fit_logit, fold_rows

_MODULE_NAME = "model_fitting"

#: Variance of the logistic noise - a constant of the model, not an estimate.
LOGISTIC_VARIANCE = math.pi**2 / 3.0

#: Normal quantile of a 95% interval: the half-width is ``HALFWIDTH_QUANTILE · SE``.
HALFWIDTH_QUANTILE = 1.96

#: The initial variance of the point shift of the GLMM fit; the coefficients start from the
#: plain logit.
SIGMA2_START = 1.0
#: How the GLMM fit obtains the variance of the point shift - the words of the passport.
SIGMA2_INSIDE_THE_FIT = "inside the fit"

_NON_FINITE_CODE = "MODEL_FITTING_NON_FINITE"


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of the fit of a passport model: estimates, their covariance, the variance of the
    point shift, sizes.

    ``separation_detected`` - the data are separated: on some direction the outcome is
    predicted without errors, and the likelihood has no interior maximum there.
    ``sigma2_covariance`` is given only by a fit with the shift of a point inside the
    likelihood: the threshold of a level depends on ``sigma2`` then, and the error of ``sigma2``
    enters the public quantities.
    """

    theta: dict[str, float]
    covariance: tuple[tuple[float, ...], ...]
    separation_detected: bool
    sigma2: float | None
    rho: float
    n_trials: int
    n_runs: int
    #: The SE of ``sigma2``; ``None`` - not estimated or on its floor (no spread).
    sigma2_se: float | None = None
    #: The covariance of ``sigma2`` with the coefficients in the order of ``theta``, then its
    #: variance; ``None`` - the variance is estimated outside the fit or not at all.
    sigma2_covariance: tuple[float, ...] | None = None
    #: How ``sigma2`` was obtained, in the words of the passport; ``None`` - no ``sigma2``.
    sigma2_method: str | None = None


def fit_vspmr_glmm(design: Design, run_weights: Mapping[str, float] | None) -> FitResult:
    """Fit a VS-PMR design with the shift of a point inside the likelihood.

    The plain logit gives the start; the GLMM fit (:mod:`iec.passport.glmm_fit`) gives the
    coefficients, their covariance - the cluster sandwich by points - and the variance of the
    shift with its SE and its covariance row. There is no fallback: when either fit does not
    converge, the passport is not computed.

    Args:
        design: Design of the passport after the exclusion rule.
        run_weights: Weight of every point by its key, or ``None`` - a plain fit.

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - the design has no trials;
            ``FIT_NOT_CONVERGED`` - a fit did not converge or gave non-finite estimates.
        ContractViolationError: ``MODEL_FITTING_NON_FINITE`` - the covariance is not a
            covariance (non-finite values or negative variances).
    """
    if design.n_trials == 0:
        raise PassportError(
            "the design has no trials: there is nothing to fit",
            error_code="PASSPORT_INPUT_MISSING",
        )
    row_weights = _row_weights(design, run_weights)
    exog, endog, groups = _arrays(design)
    rows = fold_rows(exog, endog, groups)
    run_weight = np.ones(rows.n_runs)
    if row_weights is not None:
        # All rows of a point share one weight: it is written by the point number.
        run_weight[groups] = row_weights
    start = fit_logit(rows, run_weight)
    if not start.converged or not np.all(np.isfinite(start.theta)):
        raise PassportError(
            f"the Newton fit of the passport did not converge in {start.steps} step(s); the "
            "public code has no fallback fit",
            error_code="FIT_NOT_CONVERGED",
        )
    if start.separated:
        get_logger(LogContext(module_name=_MODULE_NAME)).warning(
            "the passport fit reports perfect separation: the likelihood has no interior "
            "maximum on these data"
        )
    fit = fit_glmm(rows, run_weight, start.theta, SIGMA2_START)
    if not fit.converged or not np.all(np.isfinite(fit.theta)):
        raise PassportError(
            f"the GLMM fit of the passport did not converge in {fit.steps} step(s); the "
            "public code has no fallback fit",
            error_code="FIT_NOT_CONVERGED",
        )
    size = len(design.names)
    theta = _as_theta(design.names, fit.theta)
    covariance = _as_rows(fit.covariance[:size, :size])
    _require_sane_fit(theta, covariance)
    return FitResult(
        theta=theta,
        covariance=covariance,
        separation_detected=start.separated,
        sigma2=fit.sigma2,
        rho=fit.sigma2 / (fit.sigma2 + LOGISTIC_VARIANCE),
        n_trials=design.n_trials,
        n_runs=design.n_runs,
        sigma2_se=None if fit.on_floor else math.sqrt(float(fit.covariance[size, size])),
        sigma2_covariance=tuple(float(value) for value in fit.covariance[size]),
        sigma2_method=SIGMA2_INSIDE_THE_FIT,
    )


def _row_weights(design: Design, run_weights: Mapping[str, float] | None) -> np.ndarray | None:
    """Spread the point weights over the design rows; ``None`` stays ``None``.

    Every point of the design must have a weight: a silent one for a missing key would make
    the weighting depend on which points the caller happened to count.
    """
    if run_weights is None:
        return None
    missing = sorted(set(design.run_keys) - set(run_weights))
    if missing:
        raise ContractViolationError(
            f"run weights are missing for {len(missing)} run(s) of the design "
            f"(first: {missing[0]!r})",
            error_code="FIT_RUN_WEIGHTS_MISSING",
        )
    return np.array([run_weights[key] for key in design.run_keys], dtype=float)


def _arrays(design: Design) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the design into arrays: rows, outcomes and point numbers by first appearance."""
    exog = np.array(design.matrix, dtype=float)
    endog = np.array(design.outcomes, dtype=float)
    keys = {key: index for index, key in enumerate(dict.fromkeys(design.run_keys))}
    groups = np.array([keys[key] for key in design.run_keys], dtype=int)
    return exog, endog, groups


def _as_rows(matrix: Any) -> tuple[tuple[float, ...], ...]:
    """Return a matrix as tuples of Python floats."""
    return tuple(tuple(float(value) for value in row) for row in np.asarray(matrix))


def _as_theta(names: Sequence[str], params: Any) -> dict[str, float]:
    """Return the estimates by name as Python floats."""
    return {name: float(value) for name, value in zip(names, params, strict=True)}


def _require_sane_fit(
    theta: Mapping[str, float], covariance: tuple[tuple[float, ...], ...]
) -> None:
    """Require the fit to be numbers and a real covariance: finite, no negative variances.

    The tolerance on the diagonal follows the scale of the matrix: three products leave
    rounding of the order of the machine epsilon on a zero, and that is not a negative
    variance.

    Raises:
        ContractViolationError: ``MODEL_FITTING_NON_FINITE``.
    """
    broken: list[str] = []
    if not all(math.isfinite(value) for value in theta.values()):
        broken.append("theta")
    size = len(covariance)
    non_finite = sum(1 for row in covariance for value in row if not math.isfinite(value))
    if non_finite:
        broken.append(f"covariance ({non_finite} non-finite of {size * size})")
    elif size:
        diagonal = [covariance[i][i] for i in range(size)]
        scale = max(diagonal) if diagonal else 0.0
        tolerance = -abs(scale) * 1e-9
        negative = [i for i, value in enumerate(diagonal) if value < tolerance]
        if negative:
            names = list(theta)
            shown = ", ".join(names[i] for i in negative[:3])
            broken.append(
                f"{len(negative)} negative variances on the diagonal ({shown}"
                f"{', ...' if len(negative) > 3 else ''})"
            )
    if broken:
        raise ContractViolationError(
            f"the passport fit produced an unusable result: {', '.join(broken)}",
            error_code=_NON_FINITE_CODE,
        )
