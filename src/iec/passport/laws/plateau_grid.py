"""The law of the length × word-number grid of a quantity with a plateau.

The law is fitted in two steps: the cells of every length slice lie on the curve of a form chosen
among the candidates of the slices (``laws.grid_slices``), with shared parameters and an own level
``μ_v``, the plateau of the slice; then the law of the plateau ``μ(u)``, ``u = log₂(v/20)``, is drawn
through the plateaus of the slices. Its form is chosen by the data among the candidates of the
assignment (``laws.block_choice``): the line, the slanted hyperbola
``μ₀ + (κ_V/2)·[(u − u_*) + √((u − u_*)² + ω_V²)]``, the binary exponential and the saturation. At
vanishing load - the start of the answer (``s → 0``) of short texts (``v → 0``) - the surface tends
to the horizontal asymptote of its plateau, so the shift of the grid is that asymptote; a law of the
plateau without one has its shift at 20 words, ``u = 0``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iec.passport.laws.block_choice import BlockLaw, block_law, choose_scattered_law
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    STATUS_NOT_CONVERGED,
    delta_se,
)
from iec.passport.laws.grid_slices import (
    MIN_LEVEL_SHARE,
    SLICE_UNSEEN,
    SliceLaw,
    choose_slice_form,
    slice_value,
)
from iec.passport.trials.layout.bin_layout import BinLayout

#: Lower bound of the transition width of the plateau law.
_W_MIN = 0.001


def vs_curve(
    slice_form: LawForm,
    shared: np.ndarray,
    plateau_form: LawForm,
    plateau_params: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray:
    """Return the surface: the curve of a slice whose level is the law of the plateau ``μ(u)``.

    Args:
        slice_form: The form of the slices.
        shared: Its shared parameters (``laws.grid_slices``).
        plateau_form: The form of the law of the plateau ``μ(u)``.
        plateau_params: Its parameters.
        xu: Rows ``(ℓ, u)``, where ``ℓ = log₂ s`` is the logarithm of the word position (zero for
            the first word), ``u = log₂(v/20)``.
    """
    levels = plateau_form.value(plateau_params, xu[:, 1])
    return np.array(
        [
            float(slice_value(slice_form, shared, float(level), np.array([ell]))[0])
            for ell, level in zip(xu[:, 0], levels, strict=True)
        ]
    )


#: Fewer cells in a slice - its plateau is not computable from the points, the slice does not
#: enter the fit.
_MIN_CELLS_PER_SLICE = 2
#: Fewer determined slices - there is nothing to draw the ``C(V)`` plot from.
_MIN_SLICES = 3
#: Reason of a determined slice when the zero of the surface is undefined: no law of the plateau.
_NO_ZERO = "the plateau law is not drawn: the zero of the surface is undefined"


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

    Weighted least squares with the weights ``1/(SE(μ_v)² + τ²)`` - the own variances of the plateaus
    and their scatter around the law (``block_choice.choose_scattered_law``), as the cells of the
    slice fit (the full covariance pulls the law to one slice with a tiny error). The hyperbola starts from the largest plateau,
    ``κ_V ∈ {−0.5, −1.5}`` and ``u_* ∈ {u_max/4, u_max/2, 3·u_max/4}``; a corner parameter on a bound
    is held. The rows of the law over the raw coefficients are its sensitivity applied to the rows
    of the plateaus ``c_rows``.

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
    choice = choose_scattered_law(
        forms,
        u_values,
        c_values,
        variances,
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
    slice_forms: Sequence[LawForm],
    plateau_forms: Sequence[LawForm],
) -> dict[str, Any]:
    """Fit the grid law in two steps: the length slices, then the law of their plateaus.

    **Step 1.** The cells of a length slice lie on the curve of a form chosen among ``slice_forms``
    (``laws.grid_slices``) over ``ℓ = log₂ s``: the parameters of the curve are shared, the plateau
    ``μ_v`` is the own level of the slice. The weights are ``1/(σᵢ² + τ²)`` of the own variances of
    the cells, without their mutual dependence; the errors are carried by the full covariance
    through the sensitivity. **A slice enters only if its plateau is computable from its points**:
    at least ``_MIN_CELLS_PER_SLICE`` measured cells, at least one of them with the share of the
    level in its derivative not below ``MIN_LEVEL_SHARE``. The share is known only after a fit, so
    the slices are chosen twice: over all slices with enough cells, then without the slices whose
    plateau is not seen. With fewer than ``_MIN_SLICES`` determined slices the status is
    ``insufficient_support``.

    **Step 2.** The law of the plateau ``μ(u)``, ``u = log₂(v/20)``, is chosen through the plateaus
    of all determined slices (:func:`fit_plateau_law`). Without it the shift of the grid is
    undefined, and the status is ``insufficient_support``.

    The law of the surface is the curve of the slices with the plateau ``μ(u)``; its shift ``δ_VS``
    is the shift of the law of the plateau.

    Args:
        theta: The coefficients by name.
        covariance: Their covariance in the order of ``names``.
        names: The columns of the fit.
        abscissae: The centres of the cells: ``x`` - the word number, ``u`` - ``log₂(v/20)``.
        layout: The bin layout of the passport.
        slice_forms: The candidate forms of the slices.
        plateau_forms: The candidate forms of the law of the plateau.

    Returns:
        ``status``; ``n_points`` - the cells of step 1; ``n_parameters``; ``chi2`` - of step 1;
        ``slice_law`` - the law of the slices; ``slices_params`` - its shared parameters and
        ``slices_rows`` - their rows over the raw coefficients; ``plateau`` - the law of the plateau;
        ``slices`` - per length slice, already in the published coding (minus ``δ_VS``): ``v_bin``,
        ``words``, ``u`` (the mean of the centres of its cells), ``cells``, ``determined``; a
        determined slice has ``plateau_weight``, ``mu_v``, ``se``, ``se_fit`` (the own error of the
        plateau in the coding of the fit, without the error of ``δ_VS``), ``law`` (the law of the
        plateau at ``u`` minus ``δ_VS``), the others ``reason``. Without a law - ``status``,
        ``n_points``, ``slices`` and, if the fit failed, ``optimizer_message`` or the
        ``plateau_reason`` of the law of the plateau.
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

    # Step 1, two choices: the share of the level is known only after the first one.
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
        # The weights are the own variances of the cells with their scatter τ², without their
        # mutual dependence: the cells of one slice are strongly correlated, and generalised least
        # squares with the full covariance drove the free plateau of a slice along the common error
        # above all its cells. Errors are still carried by the full covariance - through the
        # sensitivity of the estimates to the coefficients.
        law = choose_slice_form(slice_forms, data, beta, np.diag(covariance)[idx], len(active))
        if isinstance(law, str):
            return undetermined(
                STATUS_NOT_CONVERGED, optimizer_message="no form of the slices converged"
            )
        shares = law.level_shares(data)
        invisible = {
            v: SLICE_UNSEEN
            for v in active
            if float(shares[data[:, 1] == slot[v]].max()) < MIN_LEVEL_SHARE
        }
        if not invisible:
            break
        reasons.update(invisible)
    else:
        # After the second choice the plateau vanished at another slice: no third fit is made;
        # slices without a visible plateau leave the result, the others are published by the
        # second fit.
        active = [v for v in volumes if v not in reasons]
        if len(active) < _MIN_SLICES:
            reasons.update({v: too_few for v in active})
            return undetermined(STATUS_INSUFFICIENT)
    assert isinstance(law, SliceLaw)
    sensitivity = law.rows(idx, len(order))
    width = law.width
    # Step 2: the law of the plateau through the plateaus of all determined slices; weights are
    # the own variances of the plateaus with their scatter. The shift of the grid is the shift of
    # that law.
    determined_volumes = [v for v in volumes if v not in reasons]
    plateau = fit_plateau_law(
        np.array([u_mean[v] for v in determined_volumes]),
        np.array([float(law.levels[slot[v]]) for v in determined_volumes]),
        np.array([sensitivity[width + slot[v]] for v in determined_volumes]),
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
        column = width + slot[volume]
        slices.append(
            {
                **describe(volume),
                "determined": True,
                "plateau_weight": float(shares[data[:, 1] == slot[volume]].max()),
                "mu_v": float(law.levels[slot[volume]]) - shift,
                "se": delta_se(sensitivity[column] - shift_gradient, covariance),
                # The own error of the plateau in the coding of the fit, without the shift.
                "se_fit": delta_se(sensitivity[column], covariance),
                "law": float(plateau.value(u_mean[volume])[0]) - shift,
            }
        )
    return {
        "status": STATUS_MEASURED,
        "n_points": len(selected),
        "n_parameters": len(law.fit.params),
        "chi2": float(law.fit.chi2),
        "slice_law": law,
        "slices_params": law.shared,
        "slices_rows": sensitivity[:width],
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
