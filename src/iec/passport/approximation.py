"""Approximation of the VS-PMR passport: block laws, restored bins, component correlations.

The second stage over the bins of a VS-PMR passport model. The law of the length × position
grid is fitted in two steps: the cells of every length slice lie on a hyperbola with two
asymptotes ``[(κ_S·log₂s + λ_S) + μ_v − √((κ_S·log₂s + λ_S − μ_v)² + κ_S²·ω_S²)]/2`` with
shared ``κ_S``, ``λ_S``, ``ω_S`` and an own plateau ``μ_v``; then the plateau hyperbola
``μ(u) = μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``, ``u = log₂(v/20)``, is drawn
through the plateaus of the slices. The density, table-size and rule-number laws are drawn
through the ``P``, ``M``, ``R`` bins. The shift of a block is the law at the zero of its axis:
``p → 0`` for density, ``m = 1`` and ``r = 1`` for the rules. At vanishing load - the start of
the answer (``s → 0``) of short texts (``v → 0``) - the surface tends to the horizontal
asymptote ``μ₀`` of its plateau, so its shift is ``δ_VS = μ₀``. The shifts go into the level
``α_phys``, the bins are published restored; the prediction of every trial does not change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iec.contracts import PointStatus
from iec.passport.bin_layout import BinLayout
from iec.passport.glmm_fit import marginal_threshold
from iec.passport.logit_fit import INFORMATION_RCOND
from iec.passport.measured_points import MeasuredPoint
from iec.passport.public_quantities import (
    PUBLIC_LEVELS,
    PublicInputs,
    public_quantities,
)
from iec.passport.second_stage_laws import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    STATUS_NOT_CONVERGED,
    _gls_least_squares,
    _sensitivity,
    delta_se,
    fit_p_law,
    fit_r_law,
    fit_m_law,
    increment_matrix,
)
from iec.passport.section import STATUS_EXCLUDED, _positions
from iec.passport.vspmr_model import (
    doub_bin,
    vs_grid_cells,
    vs_grid_tops,
    vspmr_density_bin,
    vspmr_density_reference,
    vspmr_density_p,
)
from iec.prompt_generator.text_generator import WORDS_PER_LINE

#: Lower bound of the transition width.
_W_MIN = 0.001
#: Relative distance to a bound at which a corner parameter of a plateau is held: the data do
#: not place it inside its range, its value is the bound and it has no error.
_HELD_TOLERANCE = 1e-6
#: Optimizer of the laws with a plateau corner. The held rule needs the fit to stop on the bound
#: the data push a corner parameter into: ``"dogbox"`` treats such a bound as active and stops
#: on it, ``"trf"`` approaches it only asymptotically. On the plateaus of the published
#: measurement the corner settled on a slice and ``"trf"`` crept to the bound of the width
#: without converging.
_HELD_METHOD = "dogbox"

#: Status of the approximation block: the laws are not measured for all blocks.
STATUS_PARTIAL = "partial"
#: Status of the approximation block: no law is measured.
STATUS_NOT_MEASURED = "not_measured"
#: Status of a restored bin: the law of its block is not measured, there is no shift.
STATUS_LAW_NOT_MEASURED = "law_not_measured"

#: Law parameters by block in the coding of the fit; the order matches the rows of the
#: sensitivity of every law. The grid law keeps the order of its earlier parameters one to one
#: (``a, b, c0, c1, W_S, u_V, W_V``), so the rows read by the public quantities stay in place;
#: ``u_star`` is ``log₂(v_*/20)``, the passport publishes the corner as ``v_star`` in words.
LAW_PARAMETERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vs_grid", ("kappa_S", "lambda_S", "mu_0", "kappa_V", "omega_S", "u_star", "omega_V")),
    ("density", ("mu_P", "A_P", "p_c")),
    ("table_size", ("lambda_M", "kappa_M")),
    ("rule_position", ("lambda_R", "A_R", "gamma")),
)
#: Components of the model in the order of the rows of the correlation matrix.
COMPONENTS: tuple[str, ...] = ("V", "S", "P", "M", "R")


def bin_abscissae(
    points: Sequence[MeasuredPoint],
    layout: BinLayout,
    run_weights: Mapping[str, float] | None,
    reference_total_words: int,
) -> dict[str, dict[str, float]]:
    """Return the centres of mass of the bins over the trials of ok points with point weights.

    Keys are the coefficient names of the model (``vs_V_S``, ``p_bin_k``, ``m_bin_k``,
    ``r_bin_k``), the reference bins included (their names are those they would have in the
    fit). Values: for a cell ``{"x": 2^(mean log₂ s), "u": ū}``, where ``s`` is the position of
    the word in the answer (``1`` for the first word) and the key ``x`` keeps the abscissa of
    the cell; the centre is geometric, as for the rules, because the law works with
    ``log₂ s``; ``u = log₂(v/reference_total_words)`` and ``reference_total_words`` is the
    reference volume of the length axis, the 20 words of the start step ``(4, 5, 1)``, not the
    first bin of the layout; for density ``{"p": p̄}``; for the
    rules ``{"m": exp(mean ln m)}`` and ``{"r": exp(mean ln r)}``. The weight of a trial is
    the weight of its point (without weights - one). A bin without trials is absent.

    The position bin of a cell is computed as in the ``trial_vector`` of a VS-PMR passport
    model: the doubling bin
    of the line number capped by the last cell of the length bin - otherwise the centres would
    fall into other cells than the coefficients.
    """
    sums: dict[str, dict[str, float]] = {}
    tops = vs_grid_tops(layout)

    def add(name: str, weight: float, **coords: float) -> None:
        entry = sums.setdefault(name, {"w": 0.0})
        entry["w"] += weight
        for key, value in coords.items():
            entry[key] = entry.get(key, 0.0) + weight * value

    for point in points:
        if point.status is not PointStatus.OK or not point.trials:
            continue
        weight = 1.0 if run_weights is None else float(run_weights.get(point.key, 1.0))
        total_words = point.D * point.n
        volume = layout.volume_bin(total_words)
        u = math.log2(total_words / reference_total_words)
        p = point.m / point.D
        k_p = vspmr_density_bin(layout, point.D, point.m)
        k_m = layout.row_bin(point.m)
        for trial in point.trials:
            position = min(doub_bin(-(-trial.word_position // WORDS_PER_LINE)), tops[volume - 1])
            add(
                f"vs_{volume}_{position}",
                weight,
                log_x=math.log2(max(1, trial.word_position)),
                u=u,
            )
            add(f"p_bin_{k_p}", weight, p=p)
            add(f"m_bin_{k_m}", weight, log_m=math.log(point.m))
            add(
                f"r_bin_{layout.row_bin(trial.rule_index)}",
                weight,
                log_r=math.log(trial.rule_index),
            )
    centres: dict[str, dict[str, float]] = {}
    for name, entry in sums.items():
        total = entry.pop("w")
        if total <= 0.0:
            continue
        mean = {key: value / total for key, value in entry.items()}
        if name.startswith("vs_"):
            centres[name] = {"x": 2.0 ** mean["log_x"], "u": mean["u"]}
        elif name.startswith("p_bin_"):
            centres[name] = {"p": mean["p"]}
        elif name.startswith("m_bin_"):
            centres[name] = {"m": math.exp(mean["log_m"])}
        else:
            centres[name] = {"r": math.exp(mean["log_r"])}
    return centres


# --- Law of the length × position grid ---------------------------------------------------


def plateau_level(mu_0: float, kappa_v: float, u_star: float, omega_v: float, u: Any) -> Any:
    """Return the plateau ``μ(u) = μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``.

    A hyperbola in ``u = log₂(v/20)``: for ``u ≪ u_*`` it tends to the horizontal asymptote
    ``μ₀`` (short texts), for ``u ≫ u_*`` to the slanted asymptote ``μ₀ + κ_V·(u − u_*)`` (long
    texts); ``u_*`` is where the asymptotes cross (the text length ``v_* = 20·2^{u_*}``), ``ω_V``
    the width of the corner, both in doublings of the text length. ``u`` may be a number or an
    array.
    """
    t = u - u_star
    return mu_0 + 0.5 * kappa_v * (t + np.sqrt(t * t + omega_v * omega_v))


def _plateau_partials(
    kappa_v: float, u_star: float, omega_v: float, u: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``∂μ/∂(μ₀, κ_V, u_*, ω_V)`` of :func:`plateau_level` at the points ``u``.

    With ``t = u − u_*`` and ``R = √(t² + ω_V²)``: ``∂μ/∂μ₀ = 1``, ``∂μ/∂κ_V = (t + R)/2``,
    ``∂μ/∂u_* = −(κ_V/2)·(1 + t/R)``, ``∂μ/∂ω_V = (κ_V/2)·ω_V/R``. ``R ≥ ω_V > 0``: the width
    has a positive lower bound.
    """
    t = u - u_star
    r = np.sqrt(t * t + omega_v * omega_v)
    return np.ones_like(t), 0.5 * (t + r), -0.5 * kappa_v * (1.0 + t / r), 0.5 * kappa_v * omega_v / r


def _vs_curve(params: np.ndarray, xu: np.ndarray) -> np.ndarray:
    """Shape of the surface: ``[(κ_S·ℓ + λ_S) + μ(u) − √((κ_S·ℓ + λ_S − μ(u))² + κ_S²·ω_S²)]/2``.

    The rows of ``xu`` are ``(ℓ, u)``, where ``ℓ = log₂ s`` is the logarithm of the word
    position (zero for the first word), ``u = log₂(v/20)``; ``params`` are
    ``(κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V)``, the plateau ``μ(u)`` is :func:`plateau_level`.
    """
    kappa_s, lambda_s, mu_0, kappa_v, omega_s, u_star, omega_v = params
    line = kappa_s * xu[:, 0] + lambda_s
    flat = plateau_level(mu_0, kappa_v, u_star, omega_v, xu[:, 1])
    gap = line - flat
    result: np.ndarray = 0.5 * (
        line + flat - np.sqrt(gap * gap + kappa_s * kappa_s * omega_s * omega_s)
    )
    return result


def vs_law_value(law: Mapping[str, Any], x: float, u: float) -> float:
    """Return the grid law at word position ``s`` and ``u`` in the coding of the fit.

    ``law`` - the law of :func:`fit_vs_law` with the parameters ``LAW_PARAMETERS["vs_grid"]``;
    the published law is in the coding with the shift and carries ``v_star`` instead of
    ``u_star``.
    """
    params = np.array([law[name] for name in dict(LAW_PARAMETERS)["vs_grid"]], dtype=float)
    return float(_vs_curve(params, np.array([[math.log2(x), u]], dtype=float))[0])


def _held(
    params: np.ndarray, lower: np.ndarray, upper: np.ndarray, candidates: Sequence[int]
) -> list[int]:
    """Return the columns among ``candidates`` whose value lies on a bound of its range.

    Such a parameter is not measured: the data push it to the edge of the range, so it is held
    there - its value is the bound, its sensitivity row is zero and it has no error.
    """
    held: list[int] = []
    for column in candidates:
        value = float(params[column])
        for bound in (float(lower[column]), float(upper[column])):
            if math.isfinite(bound) and abs(value - bound) <= _HELD_TOLERANCE * max(1.0, abs(bound)):
                held.append(column)
                break
    return held


# --- The grid law: two steps over the length slices -------------------------------------


#: Fewer cells in a slice - its plateau is not computable from the points, the slice does not
#: enter the fit.
_MIN_CELLS_PER_SLICE = 2
#: The smallest plateau share in the derivative of at least one cell of a slice: below it all
#: cells of the slice lie on the slanted part, and the plateau is not set by the data.
_MIN_PLATEAU_WEIGHT = 0.05
#: Fewer determined slices - there is nothing to draw the ``C(V)`` plot from.
_MIN_SLICES = 3
#: Fewer determined slices - the plateau law (four parameters) is not drawn through the ``C_v``:
#: the parameters plus one.
_MIN_SLICES_FOR_PLATEAU_LAW = 5
#: Reason of a determined slice when the zero of the surface is undefined: the plateau law needs
#: at least ``_MIN_SLICES_FOR_PLATEAU_LAW`` determined slices and a converged fit.
_NO_ZERO = "the plateau law is not drawn: the zero of the surface is undefined"


def _plateau_curve(params: np.ndarray, data: np.ndarray) -> np.ndarray:
    """A slice with its own plateau: ``[(κ_S·ℓ + λ_S) + μ_v − √((κ_S·ℓ + λ_S − μ_v)² + κ_S²ω_S²)]/2``.

    ``data`` - rows ``(ℓ = log₂ s, slice number 0…K−1)``; ``params`` - ``(κ_S, λ_S, ω_S,
    μ_0…μ_{K−1})``.
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
    """Return ``∂VS/∂(κ_S, λ_S, ω_S, μ_0…μ_{K−1})``: for ``μ_v`` - ``(1 + d/Q)/2`` in its cells."""
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
    """
    span = max(float(data[:, 0].max() - data[:, 0].min()), 2.0 * _W_MIN)
    top = [float(beta[data[:, 1] == slot].max()) for slot in range(count)]
    lower = np.array([-np.inf, -np.inf, _W_MIN, *([-np.inf] * count)])
    upper = np.array([0.0, np.inf, span, *([np.inf] * count)])
    start = np.array([-2.0, 0.0, 1.0, *top])
    return _gls_least_squares(
        _plateau_curve, _plateau_jacobian, start, lower, upper, data, beta, root
    )


def _plateau_law_curve(params: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The plateau law through the slice plateaus: :func:`plateau_level` with ``(μ₀, κ_V, u_*, ω_V)``."""
    result: np.ndarray = plateau_level(params[0], params[1], params[2], params[3], u)
    return result


def _plateau_law_jacobian(params: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Return ``∂μ/∂(μ₀, κ_V, u_*, ω_V)`` of the plateau law at the points ``u``."""
    return np.column_stack(_plateau_partials(params[1], params[2], params[3], u))


#: Names of the parameters of the plateau law in the order of its vector, as the passport
#: publishes them: the corner ``u_*`` is published as the text length ``v_star``. The held
#: parameters are named from here.
_PLATEAU_LAW_PARAMETERS = ("mu_0", "kappa_V", "v_star", "omega_V")


def _fit_plateau_law(
    u_values: np.ndarray, c_values: np.ndarray, c_rows: np.ndarray, covariance: np.ndarray
) -> dict[str, Any] | None:
    """Draw the plateau hyperbola through the slice plateaus ``μ_v``; ``None`` - it is not drawn.

    Weighted least squares with the weights ``1/SE(μ_v)²`` - the own variances of the plateaus,
    as the cells of the slice fit (the full covariance pulls the law to one slice with a tiny
    error). Starts: ``μ₀`` - the largest plateau, ``κ_V ∈ {−0.5, −1.5}``,
    ``u_* ∈ {u_max/4, u_max/2, 3·u_max/4}``, ``ω_V = 0.5``; bounds: ``κ_V ≤ 0``,
    ``0 ≤ u_* ≤ u_max``, ``0.001 ≤ ω_V ≤ max(u_max, 0.002)``. A corner parameter on a bound is
    held. The sensitivity of the law to the raw coefficients is ``(JᵀWJ)⁻¹JᵀW`` over the free
    columns, applied to the rows of the plateaus ``c_rows``.
    """
    if len(c_values) < _MIN_SLICES_FOR_PLATEAU_LAW:
        return None
    weights = np.array([1.0 / max(delta_se(row, covariance), 1e-12) ** 2 for row in c_rows])
    root = np.diag(np.sqrt(weights))
    u_max = max(float(u_values.max()), 2.0 * _W_MIN)
    lower = np.array([-np.inf, -np.inf, 0.0, _W_MIN])
    upper = np.array([np.inf, 0.0, u_max, u_max])
    fits = [
        _gls_least_squares(
            _plateau_law_curve,
            _plateau_law_jacobian,
            np.array([float(c_values.max()), slope, corner, 0.5]),
            lower,
            upper,
            u_values,
            c_values,
            root,
            method=_HELD_METHOD,
        )
        for slope in (-0.5, -1.5)
        for corner in (0.25 * u_max, 0.5 * u_max, 0.75 * u_max)
    ]
    result, chi2 = min(fits, key=lambda pair: pair[1])
    if not result.success:
        return None
    held = _held(result.x, lower, upper, (2, 3))
    free = [column for column in range(4) if column not in held]
    jacobian = _plateau_law_jacobian(result.x, u_values)[:, free]
    try:
        projector = np.linalg.solve(
            jacobian.T @ (weights[:, None] * jacobian), jacobian.T * weights[None, :]
        )
    except np.linalg.LinAlgError:
        return None
    sensitivity = np.zeros((4, c_rows.shape[1]), dtype=float)
    sensitivity[free] = projector @ c_rows
    return {
        "params": result.x,
        "sensitivity": sensitivity,
        "held": [_PLATEAU_LAW_PARAMETERS[column] for column in held],
        "chi2": chi2,
        "n_points": int(len(c_values)),
    }


def fit_vs_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    abscissae: Mapping[str, Mapping[str, float]],
    layout: BinLayout,
    reference_total_words: int,
) -> dict[str, Any]:
    """Fit the grid law in two steps: the length slices, then the plateau hyperbola.

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

    **Step 2.** The plateau hyperbola ``μ(u) = μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``,
    ``u = log₂(v/v_ref)``, is drawn through the plateaus of all determined slices
    (:func:`_fit_plateau_law`). Without it - fewer than ``_MIN_SLICES_FOR_PLATEAU_LAW``
    determined slices or no convergence - the shift of the grid is undefined, and the status is
    ``insufficient_support``.

    The law of the surface is the slice hyperbola with the plateau ``μ(u)``; its shift is the
    horizontal asymptote ``δ_VS = μ₀`` (:func:`_shift_vs`).

    Returns:
        ``status``; ``n_points`` - the cells of step 1; ``n_parameters`` - ``3 + K``; ``chi2`` -
        of step 1 by the own variances; the parameters ``LAW_PARAMETERS["vs_grid"]`` in the coding
        of the fit; ``v_star`` - the corner in words, ``v_ref·2^{u_*}``; ``held`` - the held corner
        parameters by their published names; ``plateau_chi2``, ``plateau_n_points`` - of step 2;
        ``sensitivity`` - the rows of ``LAW_PARAMETERS["vs_grid"]`` over the raw coefficients;
        ``slices`` - per length slice, already in the published coding (minus ``δ_VS``):
        ``v_bin``, ``words``, ``u`` (the mean of the centres of its cells), ``cells``,
        ``determined``; a determined slice has ``plateau_weight``, ``mu_v``, ``se``, ``se_fit``
        (the own error of the plateau in the coding of the fit, without the error of ``δ_VS``),
        ``law`` (the plateau hyperbola at ``u`` minus ``δ_VS``), the others ``reason``. Without a
        law - ``status``, ``n_points``, ``slices`` and, if the fit failed, ``optimizer_message``.
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
        sensitivity = _sensitivity(jacobian, precision, idx, len(order))
    except np.linalg.LinAlgError:
        return undetermined(
            STATUS_NOT_CONVERGED,
            optimizer_message="singular normal equations of the slice fit",
        )
    # Step 2: the plateau hyperbola through the plateaus of all determined slices; weights are
    # the own variances of the plateaus. The shift of the grid is its horizontal asymptote.
    determined_volumes = [v for v in volumes if v not in reasons]
    plateau_law = _fit_plateau_law(
        np.array([u_mean[v] for v in determined_volumes]),
        np.array([float(result.x[3 + slot[v]]) for v in determined_volumes]),
        np.array([sensitivity[3 + slot[v]] for v in determined_volumes]),
        covariance,
    )
    if plateau_law is None:
        reasons.update({v: _NO_ZERO for v in determined_volumes})
        return undetermined(STATUS_INSUFFICIENT)
    mu_0, kappa_v, u_star, omega_v = (float(value) for value in plateau_law["params"])
    shift_gradient = plateau_law["sensitivity"][0]
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
                "mu_v": float(result.x[column]) - mu_0,
                "se": delta_se(sensitivity[column] - shift_gradient, covariance),
                # The own error of the plateau in the coding of the fit, without the shift.
                "se_fit": delta_se(sensitivity[column], covariance),
                "law": float(plateau_level(mu_0, kappa_v, u_star, omega_v, u_mean[volume]))
                - mu_0,
            }
        )
    rows = plateau_law["sensitivity"]
    return {
        "status": STATUS_MEASURED,
        "n_points": len(selected),
        "n_parameters": len(result.x),
        "chi2": chi2,
        "kappa_S": float(result.x[0]),
        "lambda_S": float(result.x[1]),
        "mu_0": mu_0,
        "kappa_V": kappa_v,
        "omega_S": float(result.x[2]),
        "u_star": u_star,
        "omega_V": omega_v,
        "v_star": reference_total_words * 2.0**u_star,
        "held": plateau_law["held"],
        "plateau_chi2": plateau_law["chi2"],
        "plateau_n_points": plateau_law["n_points"],
        # The rows in the order of LAW_PARAMETERS["vs_grid"]: κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V.
        "sensitivity": np.vstack(
            [sensitivity[0], sensitivity[1], rows[0], rows[1], sensitivity[2], rows[2], rows[3]]
        ),
        "slices": slices,
    }


# --- Block shifts and restored bins -----------------------------------------------------


def _shift_vs(law: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """The shift of the grid ``δ_VS = μ₀`` - the horizontal asymptote of the surface.

    At vanishing load - the start of the answer (``s → 0``, the slanted asymptote of fatigue
    goes up) of short texts (``u → −∞``) - the surface tends to the horizontal asymptote ``μ₀``
    of its plateau. With ``δ_VS`` the shape ``VS − δ_VS`` tends to zero there. The gradient is
    the sensitivity row of ``μ₀``.
    """
    return float(law["mu_0"]), np.asarray(law["sensitivity"])[2]


def _shift_p(law: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """The density shift ``μ_P`` - the law at ``p → 0``; the gradient is the row of ``μ_P``."""
    return float(law["mu_P"]), np.asarray(law["sensitivity"])[0]


def _shift_m(law: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """``λ_M`` - the line at ``m = 1``."""
    return float(law["lambda_M"]), np.asarray(law["sensitivity"])[0]


def _shift_r(law: Mapping[str, Any]) -> tuple[float, np.ndarray]:
    """``λ_R`` - the curve at ``r = 1``."""
    return float(law["lambda_R"]), np.asarray(law["sensitivity"])[0]


_SHIFTS = {
    "vs_grid": _shift_vs,
    "density": _shift_p,
    "table_size": _shift_m,
    "rule_position": _shift_r,
}


def _law_at(block: str, law: Mapping[str, Any], coords: Mapping[str, float]) -> float:
    """Return the value of the block law in the fit coding (before subtracting the shift)."""
    if block == "vs_grid":
        return vs_law_value(law, coords["x"], coords["u"])
    if block == "density":
        return float(law["mu_P"]) - float(law["A_P"]) * (1.0 - math.exp(-coords["p"] / law["p_c"]))
    if block == "table_size":
        return float(law["lambda_M"]) + float(law["kappa_M"]) * math.log2(coords["m"])
    return float(law["lambda_R"]) - float(law["A_R"]) * (coords["r"] ** float(law["gamma"]) - 1.0)


def _restored_rows(
    block: str,
    entries: Sequence[tuple[dict[str, Any], str | None, Mapping[str, float] | None]],
    law: Mapping[str, Any],
    theta: Mapping[str, float],
    covariance: np.ndarray,
    order: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return the restored bins of a block: ``θ − s`` (the reference one ``−s``), law, deviation.

    ``entries`` - the row fields, the coefficient name (``None`` - the reference bin, an
    implicit zero) and the centre of mass of the bin. The law at the abscissa of a bin is in
    the same restored coding: ``law(centre) − s``. The deviation ``z = (bin − law)/SE`` is taken
    by the SE of the raw coefficient: the shift enters the bin and the law equally and cancels
    in the difference. The reference bin has no deviation: its value is a coding, not a
    measurement.
    """
    measured = law.get("status") == STATUS_MEASURED
    shift, shift_gradient = _SHIFTS[block](law) if measured else (0.0, np.zeros(0))
    rows: list[dict[str, Any]] = []
    for fields, name, coords in entries:
        row: dict[str, Any] = dict(fields)
        if block == "vs_grid":
            row["x"] = None if coords is None else coords["x"]
            row["u"] = None if coords is None else coords["u"]
        else:
            row["x"] = None if coords is None else next(iter(coords.values()))
        empty = {"value": None, "se": None, "law": None, "z": None}
        if name is not None and name not in order:
            rows.append({**row, **empty, "status": STATUS_EXCLUDED})
            continue
        if not measured:
            rows.append({**row, **empty, "status": STATUS_LAW_NOT_MEASURED})
            continue
        gradient = -shift_gradient
        raw = 0.0
        raw_se = 0.0
        if name is not None:
            gradient = gradient.copy()
            gradient[order[name]] += 1.0
            raw = float(theta[name])
            raw_se = math.sqrt(max(float(covariance[order[name], order[name]]), 0.0))
        value = raw - shift
        law_value = None if coords is None else _law_at(block, law, coords) - shift
        z = (
            (value - law_value) / raw_se
            if name is not None and law_value is not None and raw_se > 0.0
            else None
        )
        rows.append(
            {
                **row,
                "value": value,
                "se": delta_se(gradient, covariance),
                "law": law_value,
                "z": z,
                "status": STATUS_MEASURED,
            }
        )
    return rows


def _published_law(block: str, law: Mapping[str, Any], covariance: np.ndarray) -> dict[str, Any]:
    """Return the block law for the passport: status, parameters with SE, ``χ²`` - no matrices."""
    if law.get("status") != STATUS_MEASURED:
        return {
            key: law[key]
            for key in ("status", "n_points", "optimizer_message", "slices")
            if key in law
        }
    if block == "vs_grid":
        return _published_vs_law(law, covariance)
    parameters = dict(LAW_PARAMETERS)[block]
    sensitivity = np.asarray(law["sensitivity"])
    published: dict[str, Any] = {"status": law["status"], "n_points": law["n_points"]}
    if "chi2" in law:
        published["chi2"] = law["chi2"]
    for name in parameters:
        published[name] = law[name]
    published["se"] = {
        name: delta_se(sensitivity[row], covariance) for row, name in enumerate(parameters)
    }
    return published


def _published_vs_law(law: Mapping[str, Any], covariance: np.ndarray) -> dict[str, Any]:
    """Return the grid law for the passport, in the coding with the shift ``δ_VS = μ₀``.

    ``VS = δ_VS + shape``: the shape tends to zero at the start of the answer of short texts.
    ``λ_S`` is given minus ``δ_VS``; ``μ₀ − δ_VS`` is zero and is not published; the corner
    ``u_*`` is published as the text length ``v_*`` in words with the error
    ``ln 2·v_*·SE(u_*)``; ``κ_S``, ``ω_S``, ``κ_V``, ``ω_V`` do not depend on the coding. A held
    corner parameter has no error. The slices are already in this coding.
    """
    s = np.asarray(law["sensitivity"])
    shift, shift_gradient = _shift_vs(law)
    v_star = float(law["v_star"])
    values = {
        "delta_VS": (shift, shift_gradient),
        "kappa_S": (float(law["kappa_S"]), s[0]),
        "lambda_S": (float(law["lambda_S"]) - shift, s[1] - shift_gradient),
        "omega_S": (float(law["omega_S"]), s[4]),
        "kappa_V": (float(law["kappa_V"]), s[3]),
        "v_star": (v_star, math.log(2.0) * v_star * s[5]),
        "omega_V": (float(law["omega_V"]), s[6]),
    }
    published: dict[str, Any] = {
        "status": law["status"],
        "n_points": law["n_points"],
        "n_parameters": law["n_parameters"],
        "chi2": law["chi2"],
    }
    for name, (value, _) in values.items():
        published[name] = value
    published["se"] = {
        name: None if name in law["held"] else delta_se(gradient, covariance)
        for name, (_, gradient) in values.items()
    }
    published["held"] = list(law["held"])
    published["plateau_chi2"] = law["plateau_chi2"]
    published["plateau_n_points"] = law["plateau_n_points"]
    published["slices"] = law["slices"]
    return published


# --- Component correlations -------------------------------------------------------------


def _grid_increments(names: Sequence[str], layout: BinLayout, along: str) -> np.ndarray:
    """The increment matrix of the grid along length (``along="V"``) or position (``"S"``).

    A row is one increment between neighbouring MEASURED cells (both present in ``names``):
    along ``V`` - ``vs_{v+1,s} − vs_{v,s}``, along ``S`` - ``vs_{v,s+1} − vs_{v,s}``. The
    columns are all names of the fit (width ``len(names)``), so the matrix can be multiplied by
    the full covariance. A cell without a neighbour gives no increment.
    """
    order = {name: position for position, name in enumerate(names)}
    rows: list[np.ndarray] = []
    for volume, position in vs_grid_cells(layout):
        here = f"vs_{volume}_{position}"
        nxt = f"vs_{volume + 1}_{position}" if along == "V" else f"vs_{volume}_{position + 1}"
        if here in order and nxt in order:
            row = np.zeros(len(names))
            row[order[nxt]] += 1.0
            row[order[here]] -= 1.0
            rows.append(row)
    return np.array(rows) if rows else np.zeros((0, len(names)))


def _axis_increments(names: Sequence[str], block: Sequence[str | None]) -> np.ndarray:
    """The increment matrix of a one-dimensional block of width ``len(names)``.

    A bin removed by the exclusion rule drops out of the order: the increment goes through it
    to the next measured one.
    """
    order = {name: position for position, name in enumerate(names)}
    present = [name for name in block if name is None or name in order]
    if len(present) < 2:
        return np.zeros((0, len(names)))
    narrow = increment_matrix(present)
    wide = np.zeros((narrow.shape[0], len(names)))
    for column, name in enumerate(name for name in present if name is not None):
        wide[:, order[name]] = narrow[:, column]
    return wide


def component_correlations(
    covariance: np.ndarray, names: Sequence[str], layout: BinLayout
) -> dict[str, Any]:
    """The 5 × 5 matrix of canonical correlations of the components ``V, S, P, M, R``.

    By the increments: ``ρ_can = √λ_max(Σ_AA⁺ Σ_AB Σ_BB⁺ Σ_BA)`` on the covariances of the
    increments; pseudo-inversion with ``INFORMATION_RCOND``. The diagonal is ones. A component
    without increments (fewer than two measured neighbouring bins) gives ``None`` in its row
    and column.
    """
    reference = vspmr_density_reference(layout)
    increments = {
        "V": _grid_increments(names, layout, "V"),
        "S": _grid_increments(names, layout, "S"),
        "P": _axis_increments(
            names,
            [None if k == reference else f"p_bin_{k}" for k in range(1, layout.P + 1)],
        ),
        "M": _axis_increments(names, [None, *(f"m_bin_{k}" for k in range(2, layout.R + 1))]),
        "R": _axis_increments(names, [None, *(f"r_bin_{k}" for k in range(2, layout.R + 1))]),
    }
    size = len(COMPONENTS)
    canonical: list[list[float | None]] = [[None] * size for _ in COMPONENTS]
    for i, left in enumerate(COMPONENTS):
        for j, right in enumerate(COMPONENTS):
            a_left, a_right = increments[left], increments[right]
            if a_left.shape[0] == 0 or a_right.shape[0] == 0:
                continue
            if i == j:
                canonical[i][j] = 1.0
                continue
            s_aa = a_left @ covariance @ a_left.T
            s_bb = a_right @ covariance @ a_right.T
            s_ab = a_left @ covariance @ a_right.T
            product = (
                np.linalg.pinv(s_aa, rcond=INFORMATION_RCOND, hermitian=True)
                @ s_ab
                @ np.linalg.pinv(s_bb, rcond=INFORMATION_RCOND, hermitian=True)
                @ s_ab.T
            )
            largest = float(np.max(np.real(np.linalg.eigvals(product))))
            canonical[i][j] = math.sqrt(min(max(largest, 0.0), 1.0))
    return {"components": list(COMPONENTS), "canonical": canonical}


# --- The approximation block ------------------------------------------------------------

#: The name of the variance of the point shift in the extended coordinates of the public
#: quantities.
SIGMA2_NAME = "sigma2"


def _padded(vector: np.ndarray) -> np.ndarray:
    """Extend a gradient over the coefficients by a zero for ``sigma2``."""
    return np.append(np.asarray(vector, dtype=float), 0.0)


def _extended_laws(laws: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the laws with their sensitivities extended by a zero column for ``sigma2``."""
    extended: dict[str, dict[str, Any]] = {}
    for block, law in laws.items():
        sensitivity = law.get("sensitivity")
        if sensitivity is None:
            extended[block] = dict(law)
            continue
        matrix = np.asarray(sensitivity, dtype=float)
        extended[block] = {**law, "sensitivity": np.hstack([matrix, np.zeros((len(matrix), 1))])}
    return extended


def approximation_block(
    theta: Mapping[str, float],
    covariance: Sequence[Sequence[float]],
    layout: BinLayout,
    abscissae: Mapping[str, Mapping[str, float]],
    *,
    reference_total_words: int,
    sigma2_in_fit: tuple[float, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Assemble the ``approximation`` block of a VS-PMR section.

    The block laws - by generalised least squares over their Σ blocks; the shifts and
    ``α_phys`` - only when all four laws are measured; the restored bins - for every block with
    a measured law; the component correlations - with any status: they do not read the laws.

    Args:
        theta: Coefficients of the fit of the model (without those removed by the exclusion
            rule).
        covariance: Covariance of the fit in the order of ``theta``.
        layout: Layout of the passport.
        abscissae: Centres of mass of the bins (:func:`bin_abscissae`).
        reference_total_words: Reference volume of the length axis
            ``u = log₂(v/reference_total_words)`` - the words of the start step.
        sigma2_in_fit: The variance of the point shift and its covariance row with the
            coefficients, then its variance (``FitResult.sigma2_covariance``) - when the shift
            is inside the fit: the level ``q`` is then read at the threshold ``η*(q, σ²)``, and
            the error of ``σ²`` enters the public quantities. ``None`` - the shift is not in the
            fit, the threshold is the logit of ``q``.

    Returns:
        The ``approximation`` block: ``status``, ``zero_point``, ``alpha_phys``, ``shifts``,
        ``laws`` (the grid law carries its length slices - :func:`fit_vs_law`), ``restored``,
        ``public`` (a list of
        :class:`~iec.passport.public_quantities.Quantity` entries), ``component_correlations``.
    """
    names = list(theta)
    order = {name: position for position, name in enumerate(names)}
    matrix = np.array([list(row) for row in covariance], dtype=float)
    sigma = (matrix + matrix.T) / 2.0

    def column(key: str, prefix: str) -> dict[str, float]:
        return {n: v[key] for n, v in abscissae.items() if n.startswith(prefix)}

    laws: dict[str, dict[str, Any]] = {
        "vs_grid": fit_vs_law(theta, sigma, names, abscissae, layout, reference_total_words),
        "density": fit_p_law(theta, sigma, names, layout, abscissae=column("p", "p_bin_")),
        "table_size": fit_m_law(
            theta, sigma, names, layout.row_bounds, abscissae=column("m", "m_bin_")
        ),
        "rule_position": fit_r_law(
            theta, sigma, names, layout.row_bounds, abscissae=column("r", "r_bin_")
        ),
    }
    measured = [law["status"] == STATUS_MEASURED for law in laws.values()]
    status = (
        STATUS_MEASURED
        if all(measured)
        else STATUS_PARTIAL
        if any(measured)
        else STATUS_NOT_MEASURED
    )

    alpha_phys: dict[str, float] | None = None
    shifts: dict[str, dict[str, float]] | None = None
    level, level_gradient = 0.0, np.zeros(len(names))
    if status == STATUS_MEASURED:
        shifts = {}
        for block, law in laws.items():
            value, gradient = _SHIFTS[block](law)
            shifts[block] = {"value": value, "se": delta_se(gradient, sigma)}
            level += value
            level_gradient = level_gradient + gradient
        alpha_phys = {"value": level, "se": delta_se(level_gradient, sigma)}

    reference = vspmr_density_reference(layout)
    grid_entries = [
        (
            {
                "v_bin": volume,
                "s_bin": position,
                "words": list(layout.volume_bounds[volume - 1]),
                "positions": _positions(position),
            },
            f"vs_{volume}_{position}",
            abscissae.get(f"vs_{volume}_{position}"),
        )
        for volume, position in vs_grid_cells(layout)
    ]
    density_entries = [
        (
            {"bin": number, "p": list(vspmr_density_p(layout, number))},
            None if number == reference else f"p_bin_{number}",
            abscissae.get(f"p_bin_{number}"),
        )
        for number in range(1, layout.P + 1)
    ]

    def rule_entries(prefix: str, key: str) -> list[Any]:
        return [
            (
                {"bin": number, "bounds": list(bounds)},
                None if number == 1 else f"{prefix}{number}",
                {key: 1.0} if number == 1 else abscissae.get(f"{prefix}{number}"),
            )
            for number, bounds in enumerate(layout.row_bounds, start=1)
        ]

    restored = {
        "vs_grid": _restored_rows("vs_grid", grid_entries, laws["vs_grid"], theta, sigma, order),
        "density_profile": _restored_rows(
            "density", density_entries, laws["density"], theta, sigma, order
        ),
        "table_size_profile": _restored_rows(
            "table_size", rule_entries("m_bin_", "m"), laws["table_size"], theta, sigma, order
        ),
        "rule_position_profile": _restored_rows(
            "rule_position",
            rule_entries("r_bin_", "r"),
            laws["rule_position"],
            theta,
            sigma,
            order,
        ),
    }
    public: list[dict[str, Any]] | None = None
    if status == STATUS_MEASURED:
        vs_shift = _shift_vs(laws["vs_grid"])
        r_shift = _shift_r(laws["rule_position"])
        sigma2, cross = (0.0, None) if sigma2_in_fit is None else sigma2_in_fit
        size = len(names)
        extended = np.zeros((size + 1, size + 1))
        extended[:size, :size] = sigma
        if cross is not None:
            row = np.asarray(cross, dtype=float)
            extended[size, :] = row
            extended[:, size] = row
        thresholds: dict[float, tuple[float, np.ndarray]] = {}
        for q in PUBLIC_LEVELS:
            value, slope = marginal_threshold(q, sigma2)
            gradient = np.zeros(size + 1)
            gradient[size] = slope
            thresholds[q] = (value, gradient)
        inputs = PublicInputs(
            theta=theta,
            names=[*names, SIGMA2_NAME],
            covariance=extended,
            layout=layout,
            laws=_extended_laws(laws),
            alpha=level,
            alpha_gradient=_padded(level_gradient),
            vs_shift=(vs_shift[0], _padded(vs_shift[1])),
            r_shift=(r_shift[0], _padded(r_shift[1])),
            thresholds=thresholds,
        )
        public = [quantity.as_dict() for quantity in public_quantities(inputs)]
    return {
        "status": status,
        "zero_point": (
            "surface: the horizontal asymptote μ_0 of its plateau - the start of the answer "
            "(s → 0) of short texts (v → 0); p → 0, m = 1, r = 1"
        ),
        "alpha_phys": alpha_phys,
        "shifts": shifts,
        # Sensitivity matrices are not serialised: they are removed only from the published
        # copies; the shifts and the restored bins above are computed from the full laws.
        "laws": {block: _published_law(block, law, sigma) for block, law in laws.items()},
        "restored": restored,
        "public": public,
        "component_correlations": component_correlations(sigma, names, layout),
    }
