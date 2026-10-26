"""The collapse of the answer: how often a point loses the text, and on which word it does.

A point loses the text when its answer ended in a failure of the third kind (the validator):
lines dropped, merged or added in the middle of the answer, or an answer cut before the end of
the text. The collapse has two measured characteristics, and both are described by the method of
the passport model - bins, laws through the bins, the shift of every axis:

- **the rate** - the probability ψ that a point loses the text. Every answered point is one
  trial "lost / kept" - all of them, the boundary cells of the area of analysis too: the fit of
  the passport model leaves those cells out, and for the collapse they are its most telling
  part. Axes: the text length and the replacement density;
- **the place** - the probability F that a point which lost the text has lost it by the word
  ``s``. Every replacement word of a point that lost the text is one trial: its outcome is 1 if
  its line is not before the line where the answer lost the text, else 0. Axes: the word number
  ``s``, the replacement density and the share of the text before the word ``(s − 1)/v``.

The method, for every characteristic:

1. **Bins.** The logit is a level plus a coefficient of the bin of every axis. The reference
   bins are the implicit zeros. A bin whose trials all have one outcome has no estimate and goes
   out with its trials, and the pass repeats (the exclusion rule of the passport). The fit is
   the weighted logit of the passport (``fit_logit``); a point weighs inversely to the number of
   points in its cell of the run cube (``cell_balanced_weights``), the covariance is the cluster
   sandwich by points.
2. **Laws through the bins**, the reference bin not a point, by generalised least squares as
   the second stage of the passport. Bins counted from one reference bin and strongly
   correlated through it take the inverse own variances as weights, as the cells of the grid;
   the errors are carried from the full covariance. The abscissa of a bin is the centre of mass
   of its trials.
3. **Shifts.** A law at the zero of its axis - its horizontal asymptote or its value at the
   smallest load - goes into the level; the error of every number is carried from the
   covariance of the bins by the sensitivity of the laws to the bins (the delta method).

The axes that are not axes of a characteristic are checked: their bins are added to its bins
one axis at a time, and their coefficients and the Wald statistic "all of them are zero" are
published.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_distribution

from iec.contracts import PointStatus
from iec.infrastructure.exceptions import PassportError
from iec.passport.approximation import _HELD_METHOD, _W_MIN, _held, _plateau_partials, plateau_level
from iec.passport.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout
from iec.passport.logit_fit import INFORMATION_RCOND, fit_logit, fold_rows
from iec.passport.measured_points import MeasuredPoint
from iec.passport.second_stage_laws import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    STATUS_NOT_CONVERGED,
    _gls_least_squares,
    _named_points,
    _sensitivity,
    _whitening,
)
from iec.passport.vpm_cube import VpmCube, build_layout, cell_balanced_weights
from iec.passport.vspmr_model import (
    doub_bin,
    vs_grid_tops,
    vspmr_density_bin,
    vspmr_density_p,
    vspmr_density_reference,
)
from iec.prompt_generator.text_generator import WORDS_PER_LINE

#: The level of a binned fit - the logit at the reference bins.
INTERCEPT = "collapse_intercept"
#: Coefficient prefixes of the bins.
LENGTH_PREFIX = "v_bin_"
DENSITY_PREFIX = "p_bin_"
RULES_PREFIX = "m_bin_"
WORD_PREFIX = "s_bin_"
SHARE_PREFIX = "q_bin_"
#: The laws of the rate and of the place by axis, in the order of the passport.
COLLAPSE_LAWS = ("collapse_length", "collapse_density")
PLACE_LAWS = ("place_word", "place_density", "place_share")
#: The parameters of the rate law and of the place law in the order of their covariance.
RATE_LAW_PARAMETERS = ("lambda_psi", "kappa_psi", "A_psi", "gamma_psi")
PLACE_LAW_PARAMETERS = (
    "lambda_F",
    "kappa_SF",
    "s_star_F",
    "omega_SF",
    "A_F",
    "gamma_F",
    "kappa_QF",
)
#: Equal bins of the share of the text before the word.
SHARE_BINS = 10
#: Minimum points of a law - the number of its parameters plus one, as in the passport.
_MIN_POINTS_LINE = 3
_MIN_POINTS_DENSITY = 4
_MIN_POINTS_WORD = 5
#: Starts of the exponent of a density law; the best by ``χ²`` is taken.
_GAMMA_STARTS = (0.25, 0.5, 1.0)


def collapse_block(points: Sequence[MeasuredPoint]) -> dict[str, Any] | None:
    """Return the ``collapse`` block of the passport: the rate and the place of the collapse.

    Args:
        points: All published points; the answered ones (``ok`` with trials) are used.

    Returns:
        ``None`` without answered points or when the outcome does not vary - no point lost the
        text or every point did.
    """
    ok = [point for point in points if point.status is PointStatus.OK and point.trials]
    if not ok:
        return None
    lost = np.array([float(point.first_fail_kind_3_line is not None) for point in ok])
    if lost.min() == lost.max():
        return None
    weighting = cell_balanced_weights(ok, VpmCube(REFERENCE_TOTAL_WORDS))
    if weighting is None:
        return None
    layout = build_layout(ok)
    block = collapse_summary(
        length_words=np.array([float(point.D * point.n) for point in ok]),
        density=np.array([point.m / point.D for point in ok]),
        bins={
            LENGTH_PREFIX: np.array([layout.volume_bin(point.D * point.n) for point in ok]),
            DENSITY_PREFIX: np.array([vspmr_density_bin(layout, point.D, point.m) for point in ok]),
            RULES_PREFIX: np.array([layout.row_bin(point.m) for point in ok]),
        },
        lost=lost,
        weights=np.array([weighting[0][point.key] for point in ok]),
        layout=layout,
        weighting=weighting[1],
    )
    block["place"] = place_block(
        [point for point in ok if point.first_fail_kind_3_line is not None]
    )
    return block


# --- Shared machinery -------------------------------------------------------------------


def _unit(index: int, width: int) -> np.ndarray:
    """Return the unit row of one coefficient."""
    row = np.zeros(width, dtype=float)
    row[index] = 1.0
    return row


def binned_fit(
    axes: Mapping[str, np.ndarray],
    references: Mapping[str, int | None],
    outcomes: np.ndarray,
    groups: np.ndarray,
    point_weights: np.ndarray,
) -> dict[str, Any]:
    """Fit a binned logit: a level and a coefficient for every bin of the axes.

    Args:
        axes: The bin of every row by coefficient prefix, numbered from one.
        references: The reference bin of every prefix; ``None`` - the smallest bin left after
            the exclusion.
        outcomes: The outcome of every row, 0/1.
        groups: The point of every row, ``0…n−1`` - the cluster of the sandwich.
        point_weights: The weight of every point ``(n,)``.

    Returns:
        The names of the coefficients (the level first), their estimates and cluster sandwich
        covariance, the mask of the rows in the fit, the dropped bins, the reference bins.

    Raises:
        PassportError: ``FIT_NOT_CONVERGED`` - the fit did not converge.
    """
    keep = np.ones(len(outcomes), dtype=bool)
    dropped: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        for prefix, values in axes.items():
            for number in sorted(set(values[keep].tolist())):
                mask = keep & (values == number)
                share = float(outcomes[mask].mean())
                if share in (0.0, 1.0):
                    keep &= ~mask
                    dropped.append(
                        {
                            "bin": f"{prefix}{number}",
                            "trials": int(mask.sum()),
                            "outcome": "lost" if share == 1.0 else "kept",
                        }
                    )
                    changed = True
    reference: dict[str, int] = {}
    for prefix, values in axes.items():
        given = references[prefix]
        reference[prefix] = int(values[keep].min()) if given is None else int(given)
    names = [INTERCEPT]
    names += [
        f"{prefix}{number}"
        for prefix, values in axes.items()
        for number in sorted(set(values[keep].tolist()))
        if number != reference[prefix]
    ]
    column = {name: position for position, name in enumerate(names)}
    design = np.zeros((int(keep.sum()), len(names)))
    design[:, 0] = 1.0
    for prefix, values in axes.items():
        kept = values[keep]
        for number in sorted(set(kept.tolist())):
            name = f"{prefix}{number}"
            if name in column:
                design[kept == number, column[name]] = 1.0
    points, run = np.unique(groups[keep], return_inverse=True)
    rows = fold_rows(design, outcomes[keep], run.ravel())
    result = fit_logit(rows, point_weights[points])
    if not result.converged or not np.all(np.isfinite(result.theta)):
        raise PassportError(
            "a binned fit of the collapse did not converge", error_code="FIT_NOT_CONVERGED"
        )
    return {
        "names": names,
        "theta": [float(value) for value in result.theta],
        "covariance": np.asarray(result.covariance, dtype=float),
        "keep": keep,
        "dropped": dropped,
        "reference": reference,
    }


def _centres(
    prefix: str,
    bins: np.ndarray,
    values: np.ndarray,
    row_weights: np.ndarray,
    keep: np.ndarray,
    geometric: bool,
) -> dict[str, float]:
    """Return the centre of mass of every bin over its rows in the fit, by coefficient name."""
    centres: dict[str, float] = {}
    for number in sorted(set(bins[keep].tolist())):
        mask = keep & (bins == number)
        w = row_weights[mask]
        if geometric:
            centres[f"{prefix}{number}"] = float(2.0 ** (w @ np.log2(values[mask]) / w.sum()))
        else:
            centres[f"{prefix}{number}"] = float(w @ values[mask] / w.sum())
    return centres


def fit_line_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    centres: Mapping[str, float],
    prefix: str,
    abscissa: Callable[[np.ndarray], np.ndarray],
    keys: tuple[str, str],
    own_variances: bool,
) -> dict[str, Any]:
    """Fit a line ``level + slope·abscissa(x)`` through the bins of one axis.

    Args:
        theta: The coefficients of the binned fit by name.
        covariance: Their covariance.
        names: Their names in the order of the covariance.
        centres: The abscissa of every bin by name.
        prefix: The coefficient prefix of the axis.
        abscissa: The coordinate of the line from the abscissa of a bin.
        keys: The names of the level and of the slope in the passport.
        own_variances: Weights - the inverse own variances of the bins (bins strongly
            correlated through their reference bin); otherwise the inverse full covariance.

    Returns:
        The law with its ``χ²``, its points and the sensitivity of ``(level, slope)`` to the
        coefficients; the level is the line at ``abscissa = 0`` - the zero of the axis.
    """
    x, beta, block, idx, point_names = _named_points(
        theta, covariance, names, prefix, lambda number: math.nan, centres
    )
    if x.size < _MIN_POINTS_LINE:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    design = np.column_stack([np.ones_like(x), abscissa(x)])
    precision = (
        np.diag(1.0 / np.diag(block))
        if own_variances
        else np.linalg.pinv(block, rcond=INFORMATION_RCOND, hermitian=True)
    )
    coefficients = np.linalg.solve(design.T @ precision @ design, design.T @ precision @ beta)
    residual = beta - design @ coefficients
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        keys[0]: float(coefficients[0]),
        keys[1]: float(coefficients[1]),
        "chi2": float(residual @ precision @ residual),
        "points": [{"name": n, "x": float(v)} for n, v in zip(point_names, x, strict=True)],
        "sensitivity": _sensitivity(design, precision, idx, len(names)),
    }


def _power_curve(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """A density law: ``μ + A·p^γ``."""
    level, amplitude, gamma = params
    result: np.ndarray = level + amplitude * np.power(x, gamma)
    return result


def _power_jacobian(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``∂f/∂(μ, A, γ)`` of a density law."""
    _, amplitude, gamma = params
    power = np.power(x, gamma)
    return np.column_stack([np.ones_like(x), power, amplitude * power * np.log(x)])


def fit_power_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    centres: Mapping[str, float],
    prefix: str,
    keys: tuple[str, str, str],
) -> dict[str, Any]:
    """Fit a density law ``μ + A·p^γ`` through the density bins.

    Non-linear generalised least squares over the full Σ block of the bins, as the density law
    of the passport; the reference bin is not a point. ``μ`` is the curve at ``p → 0`` - its
    horizontal asymptote. Bounds: ``A ≥ 0``, ``γ > 0``. Starts: the level of the sparsest bin,
    the rise to the densest one and the exponents ``_GAMMA_STARTS``; the best by ``χ²``.

    Args:
        keys: The names of ``μ``, ``A``, ``γ`` in the passport.
    """
    x, beta, block, idx, point_names = _named_points(
        theta, covariance, names, prefix, lambda number: math.nan, centres
    )
    if x.size < _MIN_POINTS_DENSITY:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    precision, root = _whitening(block)
    lower = np.array([-np.inf, 0.0, 1e-6])
    upper = np.array([np.inf, np.inf, np.inf])
    sparse = float(beta[int(np.argmin(x))])
    rise = max(float(beta[int(np.argmax(x))]) - sparse, 0.1)
    result, chi2 = min(
        (
            _gls_least_squares(
                _power_curve,
                _power_jacobian,
                np.array([sparse, rise, gamma]),
                lower,
                upper,
                x,
                beta,
                root,
            )
            for gamma in _GAMMA_STARTS
        ),
        key=lambda pair: pair[1],
    )
    try:
        sensitivity = _sensitivity(_power_jacobian(result.x, x), precision, idx, len(names))
    except np.linalg.LinAlgError:
        sensitivity = None
    if not result.success or sensitivity is None:
        return {
            "status": STATUS_NOT_CONVERGED,
            "n_points": int(x.size),
            "optimizer_message": str(result.message),
        }
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        **{key: float(value) for key, value in zip(keys, result.x, strict=True)},
        "chi2": chi2,
        "points": [{"name": n, "x": float(v)} for n, v in zip(point_names, x, strict=True)],
        "sensitivity": sensitivity,
    }


def _public_laws(
    laws: Mapping[str, dict[str, Any]],
    parameters: Mapping[str, tuple[str, ...]],
    covariance: np.ndarray,
) -> dict[str, Any]:
    """Return the laws for the passport: without the sensitivity, with the SE of the parameters.

    A held parameter (``held``) has no SE.
    """
    public: dict[str, Any] = {}
    for key, law in laws.items():
        entry = {name: value for name, value in law.items() if name != "sensitivity"}
        if law["status"] == STATUS_MEASURED:
            sensitivity = law["sensitivity"]
            se = np.sqrt(np.clip(np.diag(sensitivity @ covariance @ sensitivity.T), 0.0, None))
            held = law.get("held") or []
            entry["se"] = {
                name: None if name in held else float(value)
                for name, value in zip(parameters[key], se.tolist(), strict=True)
            }
        public[key] = entry
    return public


def _shift(law: Mapping[str, Any], name: str, covariance: np.ndarray) -> dict[str, float]:
    """Return the shift of a block - the law at the zero of its axis - with its SE."""
    row = law["sensitivity"][0]
    return {"value": float(law[name]), "se": math.sqrt(float(row @ covariance @ row))}


def _restored(
    prefix: str,
    bounds: Sequence[tuple[float, float]],
    fit: Mapping[str, Any],
    centres: Mapping[str, float],
    shift: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Return a row per bin of an axis: its range, centre and coefficient minus the shift.

    The reference bin has the coefficient ``0`` before the shift; a dropped bin has no number.
    """
    theta = dict(zip(fit["names"], fit["theta"], strict=True))
    reference = fit["reference"][prefix]
    rows = []
    for number, (low, high) in enumerate(bounds, start=1):
        name = f"{prefix}{number}"
        if name in theta:
            value: float | None = theta[name] - float(shift["value"])
        elif number == reference:
            value = -float(shift["value"])
        else:
            value = None
        rows.append({"bin": number, "lo": low, "hi": high, "x": centres.get(name), "value": value})
    return rows


def check_bins(
    axes: Mapping[str, np.ndarray],
    references: Mapping[str, int | None],
    prefix: str,
    reference: int | None,
    bounds: Sequence[tuple[float, float]],
    outcomes: np.ndarray,
    groups: np.ndarray,
    point_weights: np.ndarray,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bins of an axis fitted together with the bins of a characteristic.

    The axis is not an axis of the characteristic; this fit checks that it is not missed.
    ``axes`` holds the axes of the characteristic and the checked one (``prefix``),
    ``references`` - the reference bins of the characteristic, ``base`` - its own binned fit.
    A row per bin of the checked axis: its range, the coefficient and its SE - ``0`` without SE
    for the reference bin, no number for a dropped bin; the Wald statistic "all coefficients of
    the axis are zero" by the cluster sandwich, with its degrees of freedom and ``p``; the
    largest change of a bin of the characteristic against ``base``, in its SE in ``base`` (the
    level is not compared: it moves with the reference bin of the checked axis).
    """
    fit = binned_fit(axes, {**references, prefix: reference}, outcomes, groups, point_weights)
    result = _check_rows(fit, prefix, bounds)
    checked = dict(zip(fit["names"], fit["theta"], strict=True))
    changes = [
        abs(checked[name] - value) / math.sqrt(float(base["covariance"][index, index]))
        for index, (name, value) in enumerate(zip(base["names"], base["theta"], strict=True))
        if name != INTERCEPT and name in checked
    ]
    result["largest_change_se"] = max(changes) if changes else None
    return result


def _check_rows(
    fit: Mapping[str, Any], prefix: str, bounds: Sequence[tuple[float, float]]
) -> dict[str, Any]:
    """Return the rows of the checked axis and the Wald statistic of its coefficients."""
    position = {name: index for index, name in enumerate(fit["names"])}
    reference = fit["reference"][prefix]
    rows = []
    for number, (low, high) in enumerate(bounds, start=1):
        name = f"{prefix}{number}"
        if name in position:
            index = position[name]
            value: float | None = fit["theta"][index]
            se: float | None = math.sqrt(float(fit["covariance"][index, index]))
        elif number == reference:
            value, se = 0.0, None
        else:
            value, se = None, None
        rows.append({"bin": number, "lo": low, "hi": high, "value": value, "se": se})
    idx = [position[name] for name in fit["names"] if name.startswith(prefix)]
    estimates = np.array([fit["theta"][index] for index in idx])
    block = fit["covariance"][np.ix_(idx, idx)]
    statistic = float(estimates @ np.linalg.solve(block, estimates)) if idx else 0.0
    return {
        "bins": rows,
        "wald": {
            "statistic": statistic,
            "df": len(idx),
            "p": float(chi2_distribution.sf(statistic, len(idx))) if idx else 1.0,
        },
    }


# --- The rate of the collapse -----------------------------------------------------------

#: The parameters of every rate law in the order of its sensitivity rows.
_RATE_PARAMETERS = {
    "collapse_length": ("lambda_Vpsi", "kappa_psi"),
    "collapse_density": ("mu_Ppsi", "A_psi", "gamma_psi"),
}


def collapse_summary(
    *,
    length_words: np.ndarray,
    density: np.ndarray,
    bins: Mapping[str, np.ndarray],
    lost: np.ndarray,
    weights: np.ndarray,
    layout: BinLayout,
    weighting: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the rate of the collapse from the outcomes of the points and their loads.

    The bins of the text length and of the density, the length law
    ``V_ψ(v) = λ_Vψ + κ_ψ·log₂(v/20)`` with the inverse own variances of the bins as weights
    (they are strongly correlated through their reference bin), the density law
    ``P_ψ(p) = μ_Pψ + A_ψ·p^γ_ψ`` over the full covariance; the shifts - the length law at
    ``v = 20``, the density law at ``p → 0`` - go into the level ``λ_ψ = α_ψ + λ_Vψ + μ_Pψ``.

    Args:
        length_words: The text length ``v`` of every point, words.
        density: The replacement density ``p = m/D`` of every point.
        bins: The bin of every point by prefix (length, density, rules), numbered from one.
        lost: 1 if the point lost the text, else 0.
        weights: Weights of the points.
        layout: The bin layout of the points.
        weighting: The summary of the weights, as ``cell_balanced_weights`` gives it.
    """
    # The length reference is the shortest length bin left after the exclusion (``None``).
    references: dict[str, int | None] = {
        LENGTH_PREFIX: None,
        DENSITY_PREFIX: vspmr_density_reference(layout),
    }
    axes = {key: bins[key] for key in references}
    groups = np.arange(len(lost))
    fit = binned_fit(axes, references, lost, groups, weights)
    names = fit["names"]
    theta = dict(zip(names, fit["theta"], strict=True))
    covariance = fit["covariance"]
    keep = fit["keep"]
    centres = {
        **_centres(LENGTH_PREFIX, bins[LENGTH_PREFIX], length_words, weights, keep, True),
        **_centres(DENSITY_PREFIX, bins[DENSITY_PREFIX], density, weights, keep, False),
    }
    laws = {
        "collapse_length": fit_line_law(
            theta,
            covariance,
            names,
            centres,
            LENGTH_PREFIX,
            lambda x: np.log2(x / REFERENCE_TOTAL_WORDS),
            ("lambda_Vpsi", "kappa_psi"),
            own_variances=True,
        ),
        "collapse_density": fit_power_law(
            theta, covariance, names, centres, DENSITY_PREFIX, ("mu_Ppsi", "A_psi", "gamma_psi")
        ),
    }
    block: dict[str, Any] = {
        "points": len(lost),
        "lost": int(lost.sum()),
        "weighting": weighting,
        "fit": _fit_payload(fit, lost),
        "rules_check": check_bins(
            {**axes, RULES_PREFIX: bins[RULES_PREFIX]},
            references,
            RULES_PREFIX,
            1,
            layout.row_bounds,
            lost,
            groups,
            weights,
            fit,
        ),
        "laws": _public_laws(laws, _RATE_PARAMETERS, covariance),
        "shifts": None,
        "restored": None,
        "rate_law": None,
    }
    if any(law["status"] != STATUS_MEASURED for law in laws.values()):
        return block
    length, density_law = laws["collapse_length"], laws["collapse_density"]
    sensitivity = np.vstack(
        [
            _unit(names.index(INTERCEPT), len(names))
            + length["sensitivity"][0]
            + density_law["sensitivity"][0],
            length["sensitivity"][1],
            density_law["sensitivity"][1],
            density_law["sensitivity"][2],
        ]
    )
    rate_theta = [
        theta[INTERCEPT] + length["lambda_Vpsi"] + density_law["mu_Ppsi"],
        length["kappa_psi"],
        density_law["A_psi"],
        density_law["gamma_psi"],
    ]
    shifts = {
        "collapse_length": _shift(length, "lambda_Vpsi", covariance),
        "collapse_density": _shift(density_law, "mu_Ppsi", covariance),
    }
    block.update(
        {
            "shifts": shifts,
            "restored": {
                "length_profile": _restored(
                    LENGTH_PREFIX, layout.volume_bounds, fit, centres, shifts["collapse_length"]
                ),
                "density_profile": _restored(
                    DENSITY_PREFIX,
                    [vspmr_density_p(layout, k) for k in range(1, layout.P + 1)],
                    fit,
                    centres,
                    shifts["collapse_density"],
                ),
            },
            "rate_law": _law_payload(RATE_LAW_PARAMETERS, rate_theta, sensitivity, covariance),
        }
    )
    return block


def _fit_payload(fit: Mapping[str, Any], outcomes: np.ndarray) -> dict[str, Any]:
    """Return the binned fit for the passport: sizes, dropped and reference bins, θ, SE, Σ."""
    covariance = fit["covariance"]
    keep = fit["keep"]
    return {
        "rows": int(keep.sum()),
        "lost": int(outcomes[keep].sum()),
        "dropped": fit["dropped"],
        "reference": fit["reference"],
        "theta": dict(zip(fit["names"], fit["theta"], strict=True)),
        "se": {name: math.sqrt(float(covariance[i, i])) for i, name in enumerate(fit["names"])},
        "covariance": covariance.tolist(),
    }


def _law_payload(
    parameters: Sequence[str],
    values: Sequence[float],
    sensitivity: np.ndarray,
    covariance: np.ndarray,
    held: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a final law: its parameters, their SE (``None`` for a held one), the covariance."""
    law_covariance = sensitivity @ covariance @ sensitivity.T
    se = np.sqrt(np.clip(np.diag(law_covariance), 0.0, None))
    return {
        "theta": dict(zip(parameters, (float(value) for value in values), strict=True)),
        "se": {
            name: None if name in held else float(value)
            for name, value in zip(parameters, se.tolist(), strict=True)
        },
        "covariance": law_covariance.tolist(),
        "held": list(held),
    }


# --- The place of the collapse ----------------------------------------------------------

#: The parameters of every place law in the order of its sensitivity rows. The corner of the
#: word law is fitted as ``u_* = log₂ s_*`` and published as the word number ``s_star_F``.
_PLACE_PARAMETERS = {
    "place_word": ("mu_SF", "kappa_SF", "s_star_F", "omega_SF"),
    "place_density": ("mu_PF", "A_F", "gamma_F"),
    "place_share": ("lambda_QF", "kappa_QF"),
}


def _word_curve(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The word law: the plateau hyperbola of the passport in ``u = log₂ s``."""
    result: np.ndarray = plateau_level(params[0], params[1], params[2], params[3], np.log2(x))
    return result


def _word_jacobian(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``∂S/∂(μ_SF, κ_SF, u_*, ω_SF)`` of the word law."""
    return np.column_stack(_plateau_partials(params[1], params[2], params[3], np.log2(x)))


def fit_word_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    centres: Mapping[str, float],
) -> dict[str, Any]:
    """Fit the word law ``S(s) = μ_SF + (κ_SF/2)·[log₂(s/s_*) + √(log₂²(s/s_*) + ω_SF²)]``.

    The hyperbola of the plateau law of the passport in the word number: the horizontal
    asymptote ``μ_SF`` at the start of the answer, the slanted one with the rise ``κ_SF`` per
    doubling of the word number past the corner ``s_*``, the width ``ω_SF`` of the corner.
    Generalised least squares over the full Σ block of the word bins, the reference bin not a
    point; the optimizer is that of the plateau law (``dogbox``), and a corner parameter on a
    bound of its range is held (``_held``): its value is the bound, its sensitivity row is zero
    and it has no error. Bounds: ``κ_SF ≥ 0``, ``0 ≤ u_* ≤ u_max``, ``0.001 ≤ ω_SF ≤ u_max``.
    Starts: ``μ_SF`` - the mean of the three first bins, ``κ_SF`` - the slope of the last four,
    ``u_* ∈ {u_max/4, u_max/2, 3·u_max/4}``, ``ω_SF = 0.5``.
    """
    x, beta, block, idx, point_names = _named_points(
        theta, covariance, names, WORD_PREFIX, lambda number: math.nan, centres
    )
    if x.size < _MIN_POINTS_WORD:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    precision, root = _whitening(block)
    u = np.log2(x)
    u_max = max(float(u.max()), 2.0 * _W_MIN)
    lower = np.array([-np.inf, 0.0, 0.0, _W_MIN])
    upper = np.array([np.inf, np.inf, u_max, u_max])
    slope = max(float(np.polyfit(u[-4:], beta[-4:], 1)[0]), 0.1)
    fits = [
        _gls_least_squares(
            _word_curve,
            _word_jacobian,
            np.array([float(beta[:3].mean()), slope, corner, 0.5]),
            lower,
            upper,
            x,
            beta,
            root,
            method=_HELD_METHOD,
        )
        for corner in (0.25 * u_max, 0.5 * u_max, 0.75 * u_max)
    ]
    result, chi2 = min(fits, key=lambda pair: pair[1])
    if not result.success:
        return {
            "status": STATUS_NOT_CONVERGED,
            "n_points": int(x.size),
            "optimizer_message": str(result.message),
        }
    held = _held(result.x, lower, upper, (2, 3))
    free = [column for column in range(4) if column not in held]
    try:
        rows = _sensitivity(_word_jacobian(result.x, x)[:, free], precision, idx, len(names))
    except np.linalg.LinAlgError:
        return {"status": STATUS_NOT_CONVERGED, "n_points": int(x.size)}
    sensitivity = np.zeros((4, len(names)), dtype=float)
    sensitivity[free] = rows
    s_star = 2.0 ** float(result.x[2])
    # The corner is published in words: SE(s_*) = ln 2 · s_* · SE(u_*).
    sensitivity[2] *= math.log(2.0) * s_star
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        "mu_SF": float(result.x[0]),
        "kappa_SF": float(result.x[1]),
        "s_star_F": s_star,
        "omega_SF": float(result.x[3]),
        "held": [_PLACE_PARAMETERS["place_word"][column] for column in held],
        "chi2": chi2,
        "points": [{"name": n, "x": float(v)} for n, v in zip(point_names, x, strict=True)],
        "sensitivity": sensitivity,
    }


def _word_bounds(top: int) -> list[tuple[int, int]]:
    """Return the words of every word bin: the doubling bins of the answer line, in words."""
    bounds = []
    for number in range(1, top + 1):
        first_line = 1 if number == 1 else 2 ** (number - 2) + 1
        last_line = 1 if number == 1 else 2 ** (number - 1)
        bounds.append(((first_line - 1) * WORDS_PER_LINE + 1, last_line * WORDS_PER_LINE))
    return bounds


def place_block(points: Sequence[MeasuredPoint]) -> dict[str, Any] | None:
    """Return the place of the collapse over the answered points that lost the text.

    Every replacement word of such a point is a trial: its outcome is 1 if its line is not
    before the line where the answer lost the text (``first_fail_kind_3_line``), else 0. The
    layout and the weights are those of the points that lost the text.

    Returns:
        ``None`` without such points or when the outcome of the words does not vary.
    """
    if not points:
        return None
    weighting = cell_balanced_weights(points, VpmCube(REFERENCE_TOTAL_WORDS))
    if weighting is None:
        return None
    layout = build_layout(points)
    tops = vs_grid_tops(layout)
    columns: dict[str, list[float]] = {
        key: [] for key in ("word", "share", "density", "lost", "group", "s", "p", "q", "v", "m")
    }
    for index, point in enumerate(points):
        line_lost = int(point.first_fail_kind_3_line or 0)
        total = point.D * point.n
        volume = layout.volume_bin(total)
        density_bin = vspmr_density_bin(layout, point.D, point.m)
        rules_bin = layout.row_bin(point.m)
        for trial in point.trials:
            line = trial.word_position // WORDS_PER_LINE + 1
            columns["word"].append(trial.word_position + 1)
            columns["share"].append(trial.word_position / total)
            columns["density"].append(point.m / point.D)
            columns["lost"].append(1.0 if line >= line_lost else 0.0)
            columns["group"].append(index)
            columns["s"].append(min(doub_bin(line), tops[volume - 1]))
            columns["p"].append(density_bin)
            columns["q"].append(min(SHARE_BINS * trial.word_position // total, SHARE_BINS - 1) + 1)
            columns["v"].append(volume)
            columns["m"].append(rules_bin)
    arrays = {key: np.asarray(values) for key, values in columns.items()}
    lost = arrays["lost"].astype(float)
    if lost.min() == lost.max():
        return None
    return place_summary(
        word=arrays["word"].astype(float),
        share=arrays["share"].astype(float),
        density=arrays["density"].astype(float),
        bins={
            WORD_PREFIX: arrays["s"].astype(int),
            DENSITY_PREFIX: arrays["p"].astype(int),
            SHARE_PREFIX: arrays["q"].astype(int),
            LENGTH_PREFIX: arrays["v"].astype(int),
            RULES_PREFIX: arrays["m"].astype(int),
        },
        lost=lost,
        groups=arrays["group"].astype(int),
        point_weights=np.array([weighting[0][point.key] for point in points]),
        layout=layout,
        weighting=weighting[1],
    )


def place_summary(
    *,
    word: np.ndarray,
    share: np.ndarray,
    density: np.ndarray,
    bins: Mapping[str, np.ndarray],
    lost: np.ndarray,
    groups: np.ndarray,
    point_weights: np.ndarray,
    layout: BinLayout,
    weighting: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the place of the collapse from the trials of the points that lost the text.

    The bins of the word number, the density and the share of the text before the word
    (``SHARE_BINS`` equal bins); the reference bins - the word bin with the most trials, the
    density bin of the start step, the first share bin. The laws: the word law
    (:func:`fit_word_law`), the density law ``P_F(p) = μ_PF + A_F·p^γ_F`` and the share line
    ``Q_F = λ_QF + κ_QF·(s − 1)/v``, all over the full covariance. The shifts - the horizontal
    asymptote ``μ_SF`` of the word law at ``s → 0``, the asymptote ``μ_PF`` at ``p → 0``, the
    share line at the start of the text ``λ_QF`` - go into the level
    ``λ_F = α_F + μ_SF + μ_PF + λ_QF``. The length and the number of rules are checked.

    Args:
        word: The word number ``s`` of every trial, from one.
        share: The share of the text before the word ``(s − 1)/v`` of every trial.
        density: The replacement density of every trial.
        bins: The bin of every trial by prefix (word, density, share, length, rules).
        lost: 1 if the text is lost by the line of the word, else 0.
        groups: The point of every trial, ``0…n−1``.
        point_weights: The weight of every point.
        layout: The bin layout of the points that lost the text.
        weighting: The summary of the weights.
    """
    counts = np.bincount(bins[WORD_PREFIX])
    references: dict[str, int | None] = {
        WORD_PREFIX: int(np.argmax(counts)),
        DENSITY_PREFIX: vspmr_density_reference(layout),
        SHARE_PREFIX: 1,
    }
    axes = {key: bins[key] for key in references}
    fit = binned_fit(axes, references, lost, groups, point_weights)
    names = fit["names"]
    theta = dict(zip(names, fit["theta"], strict=True))
    covariance = fit["covariance"]
    keep = fit["keep"]
    row_weights = point_weights[groups]
    centres = {
        **_centres(WORD_PREFIX, bins[WORD_PREFIX], word, row_weights, keep, True),
        **_centres(DENSITY_PREFIX, bins[DENSITY_PREFIX], density, row_weights, keep, False),
        **_centres(SHARE_PREFIX, bins[SHARE_PREFIX], share, row_weights, keep, False),
    }
    laws = {
        "place_word": fit_word_law(theta, covariance, names, centres),
        "place_density": fit_power_law(
            theta, covariance, names, centres, DENSITY_PREFIX, ("mu_PF", "A_F", "gamma_F")
        ),
        "place_share": fit_line_law(
            theta,
            covariance,
            names,
            centres,
            SHARE_PREFIX,
            lambda x: x,
            ("lambda_QF", "kappa_QF"),
            own_variances=False,
        ),
    }
    word_top = max(vs_grid_tops(layout))
    block: dict[str, Any] = {
        "points": int(len(point_weights)),
        "trials": int(len(lost)),
        "lost_trials": int(lost.sum()),
        "weighting": weighting,
        "fit": _fit_payload(fit, lost),
        "checks": {
            "length": check_bins(
                {**axes, LENGTH_PREFIX: bins[LENGTH_PREFIX]},
                references,
                LENGTH_PREFIX,
                None,
                layout.volume_bounds,
                lost,
                groups,
                point_weights,
                fit,
            ),
            "rules": check_bins(
                {**axes, RULES_PREFIX: bins[RULES_PREFIX]},
                references,
                RULES_PREFIX,
                1,
                layout.row_bounds,
                lost,
                groups,
                point_weights,
                fit,
            ),
        },
        "laws": _public_laws(laws, _PLACE_PARAMETERS, covariance),
        "shifts": None,
        "restored": None,
        "place_law": None,
    }
    if any(law["status"] != STATUS_MEASURED for law in laws.values()):
        return block
    word_law, density_law, share_law = (laws[key] for key in PLACE_LAWS)
    sensitivity = np.vstack(
        [
            _unit(names.index(INTERCEPT), len(names))
            + word_law["sensitivity"][0]
            + density_law["sensitivity"][0]
            + share_law["sensitivity"][0],
            word_law["sensitivity"][1],
            word_law["sensitivity"][2],
            word_law["sensitivity"][3],
            density_law["sensitivity"][1],
            density_law["sensitivity"][2],
            share_law["sensitivity"][1],
        ]
    )
    place_theta = [
        theta[INTERCEPT] + word_law["mu_SF"] + density_law["mu_PF"] + share_law["lambda_QF"],
        word_law["kappa_SF"],
        word_law["s_star_F"],
        word_law["omega_SF"],
        density_law["A_F"],
        density_law["gamma_F"],
        share_law["kappa_QF"],
    ]
    held = list(word_law["held"])
    shifts = {
        "place_word": _shift(word_law, "mu_SF", covariance),
        "place_density": _shift(density_law, "mu_PF", covariance),
        "place_share": _shift(share_law, "lambda_QF", covariance),
    }
    block.update(
        {
            "shifts": shifts,
            "restored": {
                "word_profile": _restored(
                    WORD_PREFIX, _word_bounds(word_top), fit, centres, shifts["place_word"]
                ),
                "density_profile": _restored(
                    DENSITY_PREFIX,
                    [vspmr_density_p(layout, k) for k in range(1, layout.P + 1)],
                    fit,
                    centres,
                    shifts["place_density"],
                ),
                "share_profile": _restored(
                    SHARE_PREFIX,
                    [((k - 1) / SHARE_BINS, k / SHARE_BINS) for k in range(1, SHARE_BINS + 1)],
                    fit,
                    centres,
                    shifts["place_share"],
                ),
            },
            "place_law": _law_payload(
                PLACE_LAW_PARAMETERS, place_theta, sensitivity, covariance, held
            ),
        }
    )
    return block
