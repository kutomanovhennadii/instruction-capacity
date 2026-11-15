"""The section of P_sync in the passport: model VS-PM, laws, level, capacities.

A trial is a place of the alignment, a success - the answer word stands against its reference
word. The model is ``η = VS(v, s) + P(p) + M(m) + b``, the same as the predictor model of the
purchase, and the weights of the requests equalise the cells of the cube ``V × P × M``, as for the
other quantities. The law of the grid is the law of the passport: the cells of a length slice lie
on a hyperbola with shared ``κ_S, λ_S, ω_S`` and an own plateau ``μ_v``, and the law of the plateau
``μ(u)``, ``u = log₂(v/20)``, goes through the plateaus (``laws.plateau_grid``). The forms of the law
of the plateau, of the density over ``log₂(p/2⁻¹⁰)`` and of the number of rules over ``log₂ m`` are
chosen by the data among the candidates of the assignment of forms (``config/passport_laws.yaml``,
``laws.block_choice``): the line, the slanted hyperbola, the binary exponential and the saturation,
by the smallest ``χ² + 2k``. The density and the number of rules go by generalised least squares
over the full covariance of their bins. A bin with fewer than ten requests with failures stays
outside its law.

The section goes by the steps of every section: the facts of the fit and the approximations of the
spread, the laws of the blocks, the level, the capacities by the laws and by the bins, the rows of
the bins. The shift of a block is the law at the zero of its axis: ``δ_VS`` of the plateau,
``δ_P`` of the density at ``p → 0``, ``δ_M = M(1)``; the level is ``α = δ_VS + δ_P + δ_M``. The
capacities are taken at the level 0.5, where the threshold of the logit of the median request is
zero at any variance of the shift: the instruction capacity by the law of the number of rules, the
number of rules at which ``α + [M(m) − M(1)]`` reaches zero, and by the walk along its bins - by the
law it is none when the fall of the law over its bins does not reach below zero within its 95%
interval, so the data do not bound it from above; the fatigue capacity - the word at which the full
law of the grid at ``p → 0`` and one rule reaches zero, at the shortest text holding that word,
``v = s``, by the curve of the law as for H (decision 6 of the passport standard; the first article
took it by the slanted asymptote). Instead of the guaranteed fatigue capacity the section gives how
many times full density multiplies the odds of an error, ``e^(−P̂(1))``, ``P̂(1) = P(1) − δ_P``
(decision 7). Errors are by the delta method over the coefficients of the fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import brentq

from iec.passport.fitting.chain.fit_paths import METHOD_NONE
from iec.passport.fitting.load_spread import request_loads
from iec.passport.fitting.variable_fit import VariableFit, fit_variable
from iec.passport.laws.block_choice import (
    MIN_POINTS,
    REASON_FEW_POINTS,
    BlockLaw,
    LawStack,
    block_law,
    choose_law,
    law_entry,
    unit_rows,
)
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import STATUS_MEASURED
from iec.passport.laws.law_rows import (
    LEVEL,
    LEVEL_THRESHOLD,
    capacity_row,
    fit_summary,
    half_width,
    joint_covariance,
)
from iec.passport.laws.plateau_grid import fit_vs_law, vs_curve
from iec.passport.laws.point_weights import covariance_weights
from iec.passport.laws.spread_laws import spread_forms, spread_laws
from iec.passport.quantities.steps.axis_rows import axis_rows
from iec.passport.quantities.steps.bins_walk import capacity_by_bin_walk
from iec.passport.quantities.steps.cell_rows import cell_rows
from iec.passport.quantities.steps.law_capacity import (
    no_slope_row,
    odds_factor_row,
    root_capacity_row,
)
from iec.passport.trials.design.folded_design import VariableDesign, build_design
from iec.passport.trials.layout.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout
from iec.passport.trials.place_trials import PointTrials

#: The zero of the logarithm of the density: ``u = log₂(p/P_ZERO) ≥ 0``.
P_ZERO = 2.0**-10
#: The largest binary logarithm of a capacity whose interval a floating-point number still holds.
LOG_CAPACITY_CEILING = 1000.0
#: The end of the search of a word number: ``2⁴⁰`` words.
_LOG_WORDS_TOP = 40.0
#: Status of a section whose fit or laws gave nothing.
STATUS_NO_LAWS = "no_laws"
#: The symbols of the abscissae of the chosen laws in their formulas.
_SYMBOLS = {"grid_plateau": "log₂(v/20)", "density": "log₂(p/2⁻¹⁰)", "table_size": "log₂ m"}


def sync_design(
    points: Sequence[PointTrials], layout: BinLayout, weights: Mapping[str, float]
) -> VariableDesign:
    """Return the folded matrix of model VS-PM: the cells of the grid and the bins of P and M.

    Args:
        points: The trials of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.
    """
    return build_design(points, "P_sync", layout, weights)


def _axis_law(
    design: VariableDesign,
    names: Sequence[str],
    theta: Mapping[str, float],
    sigma: np.ndarray,
    letter: str,
    forms: Sequence[LawForm],
) -> BlockLaw | str:
    """Choose the law of the density or of the number of rules by generalised least squares.

    The law goes through the bins of the axis other than the reference one with enough requests
    with failures, over the full covariance of their coefficients; the abscissa is
    ``log₂(p/2⁻¹⁰)`` for the density, whose zero lies at ``p → 0``, and ``log₂ m`` for the rules.

    Args:
        design: The folded matrix: the table of the bins of the axis and their centres.
        names: The columns of the fit.
        theta: The coefficients by name.
        sigma: The covariance of the coefficients.
        letter: The axis: ``P`` or ``M``.
        forms: The candidate forms of the block.

    Returns:
        The law, or the reason of a block without a law.
    """
    table = design.axes[letter]
    bins = [
        item.name
        for item in table.bins
        if item.number != table.reference and item.name in names
    ]
    if len(bins) < MIN_POINTS:
        return REASON_FEW_POINTS
    index = [names.index(name) for name in bins]
    key = letter.lower()
    centers = np.array([design.abscissae[name][key] for name in bins])
    x = np.log2(centers / P_ZERO) if letter == "P" else np.log2(centers)
    values = np.array([theta[name] for name in bins], dtype=float)
    choice = choose_law(forms, x, values, covariance_weights(sigma[np.ix_(index, index)]))
    if isinstance(choice, str):
        return choice
    return block_law(choice, unit_rows(index, len(names)), bins, x, at_infinity=letter == "P")


def _log_word_root(word_logit: Callable[[float], float], threshold: float) -> float:
    """Return ``log₂`` of the word at which the full law of the grid reaches the threshold.

    Args:
        word_logit: The logit of the median request at ``log₂ s`` of ``v = s`` words.
        threshold: The logit of the median request at the level of the capacity.
    """
    if word_logit(0.0) <= threshold:
        return 0.0
    if word_logit(_LOG_WORDS_TOP) >= threshold:
        return _LOG_WORDS_TOP
    return float(brentq(lambda x: word_logit(x) - threshold, 0.0, _LOG_WORDS_TOP, xtol=1e-10))


def _rule_count_root(law: BlockLaw, params: np.ndarray, alpha: float) -> float:
    """Return ``log₂ m`` at which ``α + [M(m) − M(1)]`` reaches the threshold; ``nan`` - no root.

    Args:
        law: The law of the number of rules over ``log₂ m``.
        params: Its parameters.
        alpha: The level.
    """
    start = law.shift(params)

    def logit(x: float) -> float:
        """Return the logit of the median request above the threshold at ``m = 2^x`` rules."""
        return alpha + float(law.value(x, params)[0]) - start - LEVEL_THRESHOLD

    if logit(0.0) <= 0.0:
        return 0.0
    if logit(LOG_CAPACITY_CEILING) > 0.0:
        return math.nan
    return float(brentq(logit, 0.0, LOG_CAPACITY_CEILING, xtol=1e-12))


def sync_section(
    points: Sequence[PointTrials],
    layout: BinLayout,
    weights: Mapping[str, float],
    assignment: LawAssignment,
) -> dict[str, Any]:
    """Compute the section of P_sync over the trials of the answered points.

    Args:
        points: The trials of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.
        assignment: The forms of the laws of the passport.

    Returns:
        The section: ``model``, ``weights``, ``fit``, ``status``, ``laws``, ``shifts``, ``alpha``,
        ``capacities``, ``bins``. Without a fit or without a law ``status`` is ``no_laws`` and
        only the facts of the fit are given. A fit without the shift (method ``logit``) has
        ``σ² = 0``: its numbers are not comparable with those of other measurements.
    """
    design = sync_design(points, layout, weights)
    run_weight = np.array([weights[key] for key in design.point_keys], dtype=float)
    fit = fit_variable(design, run_weight, request_loads(points, layout))
    return sync_laws(
        design, fit, layout, float(max(point.D * point.n for point in points)), assignment
    )


def sync_laws(
    design: VariableDesign,
    fit: VariableFit,
    layout: BinLayout,
    support: float,
    assignment: LawAssignment,
) -> dict[str, Any]:
    """Compute the section of P_sync over a fit of model VS-PM.

    Args:
        design: The folded matrix of the model.
        fit: Its fit.
        layout: The bin layout of the passport.
        support: The longest text of the measurement, in words.
        assignment: The forms of the laws of the passport.

    Returns:
        The section, as :func:`sync_section`. A block without a law leaves the section without
        laws: ``law_reasons`` names the block and the reason.
    """
    section: dict[str, Any] = {
        "model": "VS-PM",
        "weights": "V x P x M",
        "fit": fit_summary(fit),
        "status": STATUS_NO_LAWS,
    }
    if fit.spread is not None:
        section["spread_laws"] = spread_laws(
            section["fit"]["spread"], spread_forms(assignment, "P_sync")
        )
    if fit.method == METHOD_NONE:
        return section
    names = list(fit.names)
    theta = dict(zip(names, fit.theta, strict=True))
    joint = joint_covariance(fit)
    size = len(names)
    sigma = joint[:size, :size]
    grid = fit_vs_law(
        theta, sigma, names, design.abscissae, layout, assignment.forms_of("P_sync", "grid_plateau")
    )
    density = _axis_law(design, names, theta, sigma, "P", assignment.forms_of("P_sync", "density"))
    table = _axis_law(design, names, theta, sigma, "M", assignment.forms_of("P_sync", "table_size"))
    section["grid_status"] = grid["status"]
    section["density_status"] = STATUS_MEASURED if isinstance(density, BlockLaw) else density
    reasons = {
        block: law
        for block, law in (("density", density), ("table_size", table))
        if isinstance(law, str)
    }
    if grid["status"] != STATUS_MEASURED:
        reasons["grid_plateau"] = grid.get("plateau_reason", grid["status"])
    if reasons:
        section["law_reasons"] = reasons
        return section
    assert isinstance(density, BlockLaw) and isinstance(table, BlockLaw)
    plateau: BlockLaw = grid["plateau"]
    laws = {"plateau": plateau, "density": density, "table": table}
    stack = LawStack(
        {
            "slices": (grid["slices_params"], grid["slices_rows"]),
            **{key: (law.params, law.rows) for key, law in laws.items()},
        }
    )

    def shift_of(key: str) -> Callable[[np.ndarray], float]:
        """Return the shift of a block as a function of a vector of the stack."""
        return lambda vector: laws[key].shift(stack.part(vector, key))

    def level(vector: np.ndarray) -> float:
        """Return the level ``α = δ_VS + δ_P + δ_M`` from a vector of the stack."""
        return sum(shift_of(key)(vector) for key in laws)

    def density_value(vector: np.ndarray, p: float | None) -> float:
        """Return the density law at ``p``; ``None`` - at ``p → 0``, its shift."""
        params = stack.part(vector, "density")
        if p is None:
            return density.shift(params)
        return float(density.value(math.log2(p / P_ZERO), params)[0])

    def word_logit(vector: np.ndarray, log_s: float, p: float | None) -> float:
        """Return the logit of the median request at word ``s`` of ``v = s`` words, one rule."""
        u = max(log_s - math.log2(REFERENCE_TOTAL_WORDS), 0.0)
        surface = vs_curve(
            stack.part(vector, "slices"),
            plateau.form,
            stack.part(vector, "plateau"),
            np.array([[log_s, u]], dtype=float),
        )[0]
        return float(surface) + density_value(vector, p) + table.shift(stack.part(vector, "table"))

    def fatigue(vector: np.ndarray) -> float:
        """Return ``log₂`` of the word at which the full law at ``p → 0`` reaches the threshold."""
        return _log_word_root(lambda x: word_logit(vector, x, None), LEVEL_THRESHOLD)

    def density_change(vector: np.ndarray) -> float:
        """Return ``P̂(1)``: the change of the density law from ``p → 0`` to ``p = 1``."""
        return density_value(vector, 1.0) - density_value(vector, None)

    def instruction(vector: np.ndarray) -> float:
        """Return ``log₂`` of the instruction capacity by the law of the number of rules."""
        return _rule_count_root(table, stack.part(vector, "table"), level(vector))

    shift_rows = {key: stack.gradient(shift_of(key)) for key in laws}
    alpha = level(stack.vector)
    table_bins = design.axes["M"]
    support_rules = max(item.upper for item in table_bins.bins if item.name in table.bins)
    top = float(table.x.max())

    def fall(vector: np.ndarray) -> float:
        """Return the fall of the law of the number of rules from one rule to its last bin."""
        params = stack.part(vector, "table")
        return float(table.value(top, params)[0]) - table.shift(params)

    if fall(stack.vector) + half_width(stack.gradient(fall), joint) < 0.0:
        law_row = root_capacity_row(
            "instruction_capacity",
            instruction,
            stack,
            joint,
            float(support_rules),
            ceiling=LOG_CAPACITY_CEILING,
        )
    else:
        law_row = no_slope_row("instruction_capacity")
    m_rows = [
        (
            design.abscissae[item.name]["m"],
            0.0 if item.number == table_bins.reference else theta[item.name],
        )
        for item in table_bins.bins
        if item.number == table_bins.reference or item.name in table.bins
    ]
    capacities = [
        law_row,
        {
            "key": "instruction_capacity",
            "level": LEVEL,
            "method": "bins",
            **capacity_by_bin_walk(m_rows, table.shift(), alpha - LEVEL_THRESHOLD),
        },
        capacity_row(
            "fatigue_capacity_words",
            LEVEL,
            fatigue(stack.vector),
            half_width(stack.gradient(fatigue), joint),
            support,
        ),
        odds_factor_row(density_change, stack, joint),
    ]
    slices_rows = grid["slices_rows"]
    section.update(
        {
            "status": STATUS_MEASURED,
            "laws": {
                "grid": {
                    "kappa_S": grid["kappa_S"],
                    "lambda_S": grid["lambda_S"],
                    "omega_S": grid["omega_S"],
                    "lambda_S_restored": grid["lambda_S"] - plateau.shift(),
                    "half": {
                        "kappa_S": half_width(slices_rows[0], joint),
                        "lambda_S_restored": half_width(
                            slices_rows[1] - shift_rows["plateau"], joint
                        ),
                        "omega_S": half_width(slices_rows[2], joint),
                    },
                    "chi2": float(grid["chi2"]),
                    "n_points": int(grid["n_points"]),
                    "n_parameters": int(grid["n_parameters"]),
                },
                "grid_plateau": law_entry(plateau, joint, _SYMBOLS["grid_plateau"]),
                "density": {
                    **law_entry(density, joint, _SYMBOLS["density"]),
                    "at_one": density_value(stack.vector, 1.0),
                },
                "table_size": law_entry(table, joint, _SYMBOLS["table_size"]),
            },
            "shifts": {
                "delta_VS": {
                    "value": plateau.shift(),
                    "half": half_width(shift_rows["plateau"], joint),
                },
                "delta_P": {
                    "value": density.shift(),
                    "half": half_width(shift_rows["density"], joint),
                },
                "delta_M": {"value": table.shift(), "half": half_width(shift_rows["table"], joint)},
            },
            "alpha": {"value": alpha, "half": half_width(stack.gradient(level), joint)},
            "capacities": capacities,
            "bins": {
                "grid": cell_rows(
                    design,
                    fit,
                    plateau.shift(),
                    shift_rows["plateau"],
                    joint,
                    lambda cell, point: float(
                        vs_curve(
                            grid["slices_params"],
                            plateau.form,
                            plateau.params,
                            np.array([[math.log2(point["x"]), point["u"]]]),
                        )[0]
                    ),
                ),
                "slices": grid["slices"],
                "P": axis_rows(
                    design,
                    fit,
                    "P",
                    "p",
                    list(density.kept_bins),
                    lambda p: density_value(stack.vector, p),
                    density.shift(),
                    shift_rows["density"],
                    joint,
                ),
                "M": axis_rows(
                    design,
                    fit,
                    "M",
                    "m",
                    list(table.kept_bins),
                    lambda m: float(table.value(math.log2(m))[0]),
                    table.shift(),
                    shift_rows["table"],
                    joint,
                ),
            },
        }
    )
    return section
