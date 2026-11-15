"""The law of the length × word-number grid of a quantity with a plateau.

The law is fitted in two steps: the cells of every length slice lie on a hyperbola with two
asymptotes ``[(κ_S·log₂s + λ_S) + μ_v − √((κ_S·log₂s + λ_S − μ_v)² + κ_S²·ω_S²)]/2`` with
shared ``κ_S``, ``λ_S``, ``ω_S`` and an own plateau ``μ_v``; then the law of the plateau ``μ(u)``,
``u = log₂(v/20)``, is drawn through the plateaus of the slices. Its form is chosen by the data among
the candidates of the assignment (``laws.block_choice``): the line, the slanted hyperbola
``μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``, the binary exponential and the saturation. At
vanishing load - the start of the answer (``s → 0``) of short texts (``v → 0``) - the surface tends
to the horizontal asymptote of its plateau, so the shift of the grid is that asymptote; a law of the
plateau without one has its shift at 20 words, ``u = 0``. The slices with their own plateaus are a
joint fit of the grid and keep their formula here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iec.passport.laws.block_choice import BlockLaw, block_law, choose_law
from iec.passport.laws.point_weights import variance_weights
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    STATUS_NOT_CONVERGED,
    delta_se,
    gls_least_squares,
    gls_sensitivity,
)
from iec.passport.trials.layout.bin_layout import BinLayout

#: Lower bound of the transition width of a slice.
_W_MIN = 0.001


def vs_curve(
    slices: np.ndarray, plateau_form: LawForm, plateau_params: np.ndarray, xu: np.ndarray
) -> np.ndarray:
    """Shape of the surface: ``[(κ_S·ℓ + λ_S) + μ(u) − √((κ_S·ℓ + λ_S − μ(u))² + κ_S²·ω_S²)]/2``.

    Args:
        slices: ``(κ_S, λ_S, ω_S)`` of the slices.
        plateau_form: The form of the law of the plateau ``μ(u)``.
        plateau_params: Its parameters.
        xu: Rows ``(ℓ, u)``, where ``ℓ = log₂ s`` is the logarithm of the word position (zero for
            the first word), ``u = log₂(v/20)``.
    """
    kappa_s, lambda_s, omega_s = (float(value) for value in slices)
    line = kappa_s * xu[:, 0] + lambda_s
    flat = plateau_form.value(plateau_params, xu[:, 1])
    gap = line - flat
    result: np.ndarray = 0.5 * (
        line + flat - np.sqrt(gap * gap + kappa_s * kappa_s * omega_s * omega_s)
    )
    return result


#: Fewer cells in a slice - its plateau is not computable from the points, the slice does not
#: enter the fit.
_MIN_CELLS_PER_SLICE = 2
#: The smallest plateau share in the derivative of at least one cell of a slice: below it all
#: cells of the slice lie on the slanted part, and the plateau is not set by the data.
_MIN_PLATEAU_WEIGHT = 0.05
#: Fewer determined slices - there is nothing to draw the ``C(V)`` plot from.
_MIN_SLICES = 3
#: Reason of a determined slice when the zero of the surface is undefined: no law of the plateau.
_NO_ZERO = "the plateau law is not drawn: the zero of the surface is undefined"


def _plateau_curve(params: np.ndarray, data: np.ndarray) -> np.ndarray:
    """A slice with its own plateau: ``[(κ_S·ℓ + λ_S) + μ_v − √((κ_S·ℓ + λ_S − μ_v)² + κ_S²ω_S²)]/2``.

    Args:
        params: ``(κ_S, λ_S, ω_S, μ_0…μ_{K−1})``.
        data: Rows ``(ℓ = log₂ s, slice number 0…K−1)``.
    """
    kappa_s, lambda_s, omega_s = params[:3]
    plateau = params[3:][data[:, 1].astype(int)]
    line = kappa_s * data[:, 0] + lambda_s
    gap = line - plateau
    result: np.ndarray = 0.5 * (
        line + plateau - np.sqrt(gap * gap + kappa_s * kappa_s * omega_s * omega_s)
    )
    return result


def _plateau_jacobian(params: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Return ``∂VS/∂(κ_S, λ_S, ω_S, μ_0…μ_{K−1})``: for ``μ_v`` - ``(1 + d/Q)/2`` in its cells.

    Args:
        params: The parameters of :func:`_plateau_curve`.
        data: Its rows.
    """
    kappa_s, lambda_s, omega_s = params[:3]
    slices = data[:, 1].astype(int)
    log_x = data[:, 0]
    d = kappa_s * log_x + lambda_s - params[3:][slices]
    q = np.sqrt(d * d + kappa_s * kappa_s * omega_s * omega_s)
    jacobian = np.zeros((len(log_x), len(params)))
    jacobian[:, 0] = 0.5 * (log_x - (d * log_x + kappa_s * omega_s * omega_s) / q)
    jacobian[:, 1] = 0.5 * (1.0 - d / q)
    jacobian[:, 2] = -0.5 * kappa_s * kappa_s * omega_s / q
    jacobian[np.arange(len(log_x)), 3 + slices] = 0.5 * (1.0 + d / q)
    return jacobian


def _fit_plateaus(
    data: np.ndarray, beta: np.ndarray, root: np.ndarray, count: int
) -> tuple[Any, float]:
    """Fit the slices with shared ``κ_S``, ``λ_S``, ``ω_S`` and own ``μ_v``; return the solution, ``χ²``.

    The start: ``κ_S = −2``, ``λ_S = 0``, ``ω_S = 1``, ``μ_v`` - the top cell of the slice.
    Bounds: ``κ_S ≤ 0``, ``0.001 ≤ ω_S ≤`` the span of ``log₂ s``.

    Args:
        data: The rows of the cells (:func:`_plateau_curve`).
        beta: The coefficients of the cells.
        root: The root of the weights of the cells.
        count: The number of slices.
    """
    span = max(float(data[:, 0].max() - data[:, 0].min()), 2.0 * _W_MIN)
    top = [float(beta[data[:, 1] == slot].max()) for slot in range(count)]
    lower = np.array([-np.inf, -np.inf, _W_MIN, *([-np.inf] * count)])
    upper = np.array([0.0, np.inf, span, *([np.inf] * count)])
    start = np.array([-2.0, 0.0, 1.0, *top])
    return gls_least_squares(
        _plateau_curve, _plateau_jacobian, start, lower, upper, data, beta, root
    )


def _plateau_starts(u_values: np.ndarray, c_values: np.ndarray) -> list[np.ndarray]:
    """Return the starts of the plateau hyperbola: the largest plateau, two slopes, three corners."""
    u_max = max(float(u_values.max()), 2.0 * _W_MIN)
    return [
        np.array([float(c_values.max()), slope, corner, 0.5])
        for slope in (-0.5, -1.5)
        for corner in (0.25 * u_max, 0.5 * u_max, 0.75 * u_max)
    ]


def fit_plateau_law(
    u_values: np.ndarray,
    c_values: np.ndarray,
    c_rows: np.ndarray,
    covariance: np.ndarray,
    forms: Sequence[LawForm],
) -> BlockLaw | str:
    """Choose the law of the plateau through the slice plateaus ``μ_v``.

    Weighted least squares with the weights ``1/SE(μ_v)²`` - the own variances of the plateaus,
    as the cells of the slice fit (the full covariance pulls the law to one slice with a tiny
    error). The hyperbola starts from the largest plateau, ``κ_V ∈ {−0.5, −1.5}`` and
    ``u_* ∈ {u_max/4, u_max/2, 3·u_max/4}``; a corner parameter on a bound is held. The rows of the
    law over the raw coefficients are its sensitivity applied to the rows of the plateaus ``c_rows``.

    Args:
        u_values: The centres of the determined slices in ``log₂(v/20)``.
        c_values: Their plateaus.
        c_rows: The rows of the plateaus over the coefficients.
        covariance: The covariance of the coefficients.
        forms: The candidate forms of the plateau.

    Returns:
        The law, or the reason of a plateau without a law.
    """
    variances = np.array([max(delta_se(row, covariance), 1e-12) ** 2 for row in c_rows])
    choice = choose_law(
        forms,
        u_values,
        c_values,
        variance_weights(variances),
        starts={"slanted_hyperbolic": _plateau_starts(u_values, c_values)},
    )
    if isinstance(choice, str):
        return choice
    names = [f"slice_{slot}" for slot in range(len(u_values))]
    return block_law(choice, c_rows, names, u_values, at_infinity=True)


def fit_vs_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    abscissae: Mapping[str, Mapping[str, float]],
    layout: BinLayout,
    plateau_forms: Sequence[LawForm],
) -> dict[str, Any]:
    """Fit the grid law in two steps: the length slices, then the law of their plateaus.

    **Step 1.** The cells of a length slice lie on the slice hyperbola
    ``[(κ_S·ℓ + λ_S) + μ_v − √((κ_S·ℓ + λ_S − μ_v)² + κ_S²·ω_S²)]/2``, ``ℓ = log₂ s``: the slope
    ``κ_S``, the level ``λ_S`` of the slanted asymptote and the width ``ω_S`` are shared, the
    plateau ``μ_v`` is the own one of the slice. The weights are the own variances of the cells,
    without their mutual dependence; the errors are carried by the full covariance through the
    sensitivity. **A slice enters only if its plateau is computable from its points**: at least
    ``_MIN_CELLS_PER_SLICE`` measured cells, at least one of them with the plateau share
    ``(1 + d/Q)/2`` of its derivative not below ``_MIN_PLATEAU_WEIGHT``. The share is known only
    after a fit, so the fit is made twice: over all slices with enough cells, then without the
    slices whose plateau is not seen. With fewer than ``_MIN_SLICES`` determined slices the
    status is ``insufficient_support``.

    **Step 2.** The law of the plateau ``μ(u)``, ``u = log₂(v/20)``, is chosen through the plateaus
    of all determined slices (:func:`fit_plateau_law`). Without it the shift of the grid is
    undefined, and the status is ``insufficient_support``.

    The law of the surface is the slice hyperbola with the plateau ``μ(u)``; its shift ``δ_VS`` is
    the shift of the law of the plateau.

    Args:
        theta: The coefficients by name.
        covariance: Their covariance in the order of ``names``.
        names: The columns of the fit.
        abscissae: The centres of the cells: ``x`` - the word number, ``u`` - ``log₂(v/20)``.
        layout: The bin layout of the passport.
        plateau_forms: The candidate forms of the law of the plateau.

    Returns:
        ``status``; ``n_points`` - the cells of step 1; ``n_parameters`` - ``3 + K``; ``chi2`` - of
        step 1 by the own variances; ``slices_params`` - ``(κ_S, λ_S, ω_S)`` and ``slices_rows`` -
        their rows over the raw coefficients; ``plateau`` - the law of the plateau; ``slices`` - per
        length slice, already in the published coding (minus ``δ_VS``): ``v_bin``, ``words``, ``u``
        (the mean of the centres of its cells), ``cells``, ``determined``; a determined slice has
        ``plateau_weight``, ``mu_v``, ``se``, ``se_fit`` (the own error of the plateau in the coding
        of the fit, without the error of ``δ_VS``), ``law`` (the law of the plateau at ``u`` minus
        ``δ_VS``), the others ``reason``. Without a law - ``status``, ``n_points``, ``slices`` and,
        if the fit failed, ``optimizer_message`` or the ``plateau_reason`` of the law of the plateau.
    """
    order = list(names)
    measured = [
        name
        for name in order
        if name.startswith("vs_")
        and name in abscissae
        and covariance[order.index(name), order.index(name)] > 0.0
    ]
    by_volume: dict[int, list[str]] = {}
    for name in measured:
        by_volume.setdefault(int(name.split("_")[1]), []).append(name)
    volumes = sorted(by_volume)
    u_mean = {v: float(np.mean([abscissae[n]["u"] for n in by_volume[v]])) for v in volumes}
    reasons: dict[int, str] = {
        v: "fewer than two cells" for v in volumes if len(by_volume[v]) < _MIN_CELLS_PER_SLICE
    }

    def describe(volume: int) -> dict[str, Any]:
        return {
            "v_bin": volume,
            "words": list(layout.volume_bounds[volume - 1]),
            "u": u_mean[volume],
            "cells": len(by_volume[volume]),
        }

    def undetermined(status: str, **extra: Any) -> dict[str, Any]:
        return {
            "status": status,
            "n_points": len(measured),
            "slices": [
                {**describe(v), "determined": False, "reason": reasons.get(v, "the fit failed")}
                for v in volumes
            ],
            **extra,
        }

    too_few = f"fewer than {_MIN_SLICES} determined slices"

    # Step 1, two fits: the plateau share is known only after the first one.
    for _ in range(2):
        active = [v for v in volumes if v not in reasons]
        if len(active) < _MIN_SLICES:
            reasons.update({v: too_few for v in active})
            return undetermined(STATUS_INSUFFICIENT)
        selected = [n for v in active for n in by_volume[v]]
        slot = {v: k for k, v in enumerate(active)}
        data = np.array(
            [[math.log2(abscissae[n]["x"]), slot[int(n.split("_")[1])]] for n in selected],
            dtype=float,
        )
        beta = np.array([float(theta[n]) for n in selected], dtype=float)
        idx = [order.index(n) for n in selected]
        # Weights are the own variances of the cells, without their mutual dependence: the
        # cells of one slice are strongly correlated, and generalised least squares with the
        # full covariance drove the free plateau of a slice along the common error above all
        # its cells. Errors are still carried by the full covariance - through the
        # sensitivity of the estimates to the coefficients.
        variances = np.diag(covariance)[idx]
        precision = np.diag(1.0 / variances)
        root = np.diag(1.0 / np.sqrt(variances))
        result, chi2 = _fit_plateaus(data, beta, root, len(active))
        if not result.success:
            return undetermined(STATUS_NOT_CONVERGED, optimizer_message=str(result.message))
        jacobian = _plateau_jacobian(result.x, data)
        invisible = {
            v: "all cells on the slope: the plateau is not seen"
            for v in active
            if float(jacobian[data[:, 1] == slot[v], 3 + slot[v]].max()) < _MIN_PLATEAU_WEIGHT
        }
        if not invisible:
            break
        reasons.update(invisible)
    else:
        # After the second fit the plateau vanished at another slice: no third fit is made;
        # slices without a visible plateau leave the result, the others are published by the
        # second fit.
        active = [v for v in volumes if v not in reasons]
        if len(active) < _MIN_SLICES:
            reasons.update({v: too_few for v in active})
            return undetermined(STATUS_INSUFFICIENT)
    try:
        sensitivity = gls_sensitivity(jacobian, precision, idx, len(order))
    except np.linalg.LinAlgError:
        return undetermined(
            STATUS_NOT_CONVERGED,
            optimizer_message="singular normal equations of the slice fit",
        )
    # Step 2: the law of the plateau through the plateaus of all determined slices; weights are
    # the own variances of the plateaus. The shift of the grid is the shift of that law.
    determined_volumes = [v for v in volumes if v not in reasons]
    plateau = fit_plateau_law(
        np.array([u_mean[v] for v in determined_volumes]),
        np.array([float(result.x[3 + slot[v]]) for v in determined_volumes]),
        np.array([sensitivity[3 + slot[v]] for v in determined_volumes]),
        covariance,
        plateau_forms,
    )
    if isinstance(plateau, str):
        reasons.update({v: _NO_ZERO for v in determined_volumes})
        return undetermined(STATUS_INSUFFICIENT, plateau_reason=plateau)
    shift = plateau.shift()
    shift_gradient = np.zeros(len(order))
    shift_gradient[:] = _shift_row(plateau)
    slices: list[dict[str, Any]] = []
    for volume in volumes:
        if volume in reasons:
            slices.append({**describe(volume), "determined": False, "reason": reasons[volume]})
            continue
        column = 3 + slot[volume]
        slices.append(
            {
                **describe(volume),
                "determined": True,
                "plateau_weight": float(jacobian[data[:, 1] == slot[volume], column].max()),
                "mu_v": float(result.x[column]) - shift,
                "se": delta_se(sensitivity[column] - shift_gradient, covariance),
                # The own error of the plateau in the coding of the fit, without the shift.
                "se_fit": delta_se(sensitivity[column], covariance),
                "law": float(plateau.value(u_mean[volume])[0]) - shift,
            }
        )
    return {
        "status": STATUS_MEASURED,
        "n_points": len(selected),
        "n_parameters": len(result.x),
        "chi2": chi2,
        "kappa_S": float(result.x[0]),
        "lambda_S": float(result.x[1]),
        "omega_S": float(result.x[2]),
        "slices_params": np.array(result.x[:3], dtype=float),
        "slices_rows": np.vstack([sensitivity[0], sensitivity[1], sensitivity[2]]),
        "plateau": plateau,
        "slices": slices,
    }


def _shift_row(plateau: BlockLaw) -> np.ndarray:
    """Return the derivative of the shift of the law of the plateau by the raw coefficients."""
    params = plateau.params
    by_parameter = np.zeros(params.size)
    for column in range(params.size):
        step = 1e-5 * max(abs(float(params[column])), 1.0)
        unit = np.zeros(params.size)
        unit[column] = step
        by_parameter[column] = (plateau.shift(params + unit) - plateau.shift(params - unit)) / (
            2.0 * step
        )
    result: np.ndarray = by_parameter @ plateau.rows
    return result
