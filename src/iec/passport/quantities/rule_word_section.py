"""The section of H in the passport: model VS-PMR, laws, level, capacities.

A trial is a rule word standing on its place, a success - the right replacement. The model is
``η = VS(v, s) + P(p) + M(m) + R(r) + b``. A slice of the grid is the curve of a form chosen among
the candidates of the slices over ``x = log₂ s`` with an own level ``μ_v`` at ``s → 0``
(``laws.grid_slices``); the levels of the slices lie on the law of the levels over
``u = log₂(v/20)``. The weights of these two fits are ``1/(σᵢ² + τ²)`` of the own variances, without
the mutual dependence: with the full covariance the curve of a slice runs away from its cells. The
other laws go with the weights ``1/(σᵢ² + τ²)`` of the own variances of their bins: the number of
rules over ``log₂ m``, the density over ``log₂(p/2⁻¹⁰)`` and the rule number over ``log₂ r``. The
form of every law is chosen by the data among the candidates of the assignment of forms
(``config/passport_laws.yaml``, ``laws.block_choice``) by the smallest ``QAIC``. The cells of the
grid are not selected.

The section goes by the steps of every section: the facts of the fit and the approximations of the
spread, the laws of the blocks, the level, the capacities by the laws and by the bins, the rows of
the bins. The shift of a block is its law at the zero of its axis: 20 words and ``s → 0`` for the
grid (``μ₀``, the law of the levels at ``u = 0``), ``p = 2⁻¹⁰`` for the density (``μ_P``: its two-slope
hyperbola has no horizontal asymptote at ``p → 0``), ``m = 1`` and
``r = 1`` for the rules; the level is ``α = μ₀ + μ_P + λ_M + λ_R``. The capacities follow the
article, sections 4.5 and 5.5, at the level 0.5, where the threshold of the logit of the median
request is zero at any variance of the shift: the instruction capacity is the root of the cumulative
price of the table ``C(m) = [M(1) − M(m)] + ∫₁^m [R(1) − R(r)] dr`` at the reserve ``α``, also by the
bins. The slices have no common asymptote, so the fatigue capacity is by the full law of the grid at
the shortest text holding the word, ``v = s``. The section publishes the parameters of its laws with
their covariance, ``law_covariance``, from which the standard stress test of the passport takes its
errors (``iec.passport.stress_test``). Errors are by the delta method over the coefficients of the fit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from iec.passport.fitting.chain.fit_paths import METHOD_NONE
from iec.passport.fitting.load_spread import request_loads
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.block_choice import (
    MIN_POINTS,
    REASON_FEW_POINTS,
    BlockLaw,
    LawStack,
    block_law,
    choose_scattered_law,
    law_entry,
    unit_rows,
)
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import STATUS_MEASURED
from iec.passport.laws.grid_slices import SliceLaw, choose_slice_form, slice_entry, slice_value
from iec.passport.laws.law_rows import (
    HALF,
    LEVEL,
    LEVEL_THRESHOLD,
    capacity_row,
    fit_summary,
    half_width,
    joint_covariance,
    own_variances,
)
from iec.passport.laws.spread_laws import spread_forms, spread_laws
from iec.passport.quantities.steps.axis_rows import axis_rows
from iec.passport.quantities.steps.bins_walk import STATUS_BELOW_ONE_RULE, STATUS_BEYOND_SUPPORT
from iec.passport.quantities.steps.cell_rows import cell_rows
from iec.passport.quantities.steps.median_check import axis_bins, checked_fit, requests_of
from iec.passport.quantities.steps.section_fit import SectionFit, light_design
from iec.passport.trials.design.folded_design import AxisBin, GridCell, VariableDesign, build_design
from iec.passport.trials.layout.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout
from iec.passport.trials.place_trials import PointTrials

#: The zero of the logarithm of the density and the logarithm of ``p = 1`` from it.
P_ZERO = 2.0**-10
_U_AT_ONE = 10.0
#: The grid of ``log₂ m`` of the cumulative price: from one rule to ``2⁴⁰`` rules.
_X_GRID = np.linspace(0.0, 40.0, 8001)
#: The end of the search of the fatigue root, ``2⁶⁰`` words: past it the law has no root.
_X_ROOT_TOP = 60.0
#: The farthest left the search of the fatigue root goes: a curve still above the threshold there
#: has no root.
_X_ROOT_FLOOR = 1e4
STATUS_NO_LAWS = "no_laws"
STATUS_NO_ROOT = "no_root"
#: The symbols of the abscissae of the chosen laws in their formulas.
_SYMBOLS = {
    "grid_plateau": "log₂(v/20)",
    "density": "log₂(p/2⁻¹⁰)",
    "table_size": "log₂ m",
    "rule_position": "log₂ r",
}


@dataclass(frozen=True, slots=True, eq=False)
class _GridLaw:
    """The slices of the grid of H: the law of the slices and the levels the law of the levels reads.

    ``cells`` - the measured cells; ``cell_index`` - their columns in the fit; ``slices`` - the
    length bins in order; ``law`` - the law of the slices; ``level_variances`` - the own variances of
    the levels; ``slice_u`` - the centres of the slices in ``log₂(v/20)``.
    """

    cells: list[GridCell]
    cell_index: list[int]
    slices: list[int]
    law: SliceLaw
    level_variances: np.ndarray
    slice_u: list[float]

    def level_rows(self, size: int) -> np.ndarray:
        """Return the derivative of the levels by all the coefficients of the fit."""
        return self.law.rows(self.cell_index, size)[self.law.width :]

    def shared_rows(self, size: int) -> np.ndarray:
        """Return the derivative of the shared parameters by all the coefficients of the fit."""
        return self.law.rows(self.cell_index, size)[: self.law.width]


def _axis_items(design: VariableDesign, fit: VariableFit, letter: str) -> list[AxisBin]:
    """Return the bins of the law of an axis: measured, not the reference bin.

    Args:
        design: The folded matrix: the table of the bins of the axis.
        fit: The fit of the quantity.
        letter: The axis: ``P``, ``M`` or ``R``.
    """
    table = design.axes[letter]
    return [
        item
        for item in table.bins
        if item.number != table.reference and item.name in fit.names
    ]


def log_capacity(
    table: BlockLaw,
    table_params: np.ndarray,
    rule: BlockLaw,
    rule_params: np.ndarray,
    reserve: float,
) -> float:
    """Return ``log₂ m``: the root of the cumulative price; no reserve - a negative number by M.

    The price of a table of ``m`` rules is ``[M(1) − M(m)] + ∫₁^m [R(1) − R(r)] dr``; the root is its
    first crossing of the reserve on the grid ``_X_GRID``, and a price that never reaches the reserve
    gives the end of the grid. Without a reserve the number is the reserve over the fall of M in its
    first doubling.

    Args:
        table: The law of the number of rules over ``log₂ m``.
        table_params: Its parameters.
        rule: The law of the rule number over ``log₂ r``.
        rule_params: Its parameters.
        reserve: The reserve of the level.
    """
    lambda_m = table.shift(table_params)
    if reserve <= 0.0:
        fall = float(table.value(1.0, table_params)[0]) - lambda_m
        return reserve / max(abs(fall), 1e-12)
    lambda_r = rule.shift(rule_params)
    integrand = (lambda_r - rule.value(_X_GRID, rule_params)) * np.log(2.0) * 2.0**_X_GRID
    steps = np.diff(_X_GRID)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * steps)])
    price = lambda_m - table.value(_X_GRID, table_params) + cumulative
    return _first_crossing(price, _X_GRID, reserve)


def _first_crossing(curve: np.ndarray, grid: np.ndarray, level: float) -> float:
    """Return the abscissa where a curve first reaches a level, linear between the grid points.

    A curve that does not reach the level gives the end of the grid; a curve already at the level at
    the start gives the start.
    """
    above = np.nonzero(curve >= level)[0]
    if above.size == 0:
        return float(grid[-1])
    k = int(above[0])
    if k == 0:
        return float(grid[0])
    low, high = float(curve[k - 1]), float(curve[k])
    return float(grid[k - 1] + (level - low) / (high - low) * (grid[k] - grid[k - 1]))


def log_fatigue(
    slice_form: LawForm,
    shared: np.ndarray,
    plateau: BlockLaw,
    plateau_params: np.ndarray,
    alpha: float,
    threshold: float,
) -> float | None:
    """Return ``log₂ s`` at ``v = s`` by the full law of the grid; ``None`` - the law has no root.

    Texts below 20 words add no price of the length, so at ``v = s`` the logit is
    ``α + [S(x; L(max(x − log₂ 20, 0))) − L(0)]``: ``S(x; μ)`` the curve of a slice of the level ``μ``
    at ``x = log₂ s`` and ``L`` the law of the levels of the slices over ``log₂(v/20)``. At ``s → 0``
    the curve tends to its level and the logit to ``α``, so a level at or below the threshold has no
    root; a logit still above it at ``2⁶⁰`` words has none either.

    Args:
        slice_form: The form of the slices.
        shared: Its shared parameters.
        plateau: The law of the levels of the slices.
        plateau_params: Its parameters.
        alpha: The level.
        threshold: The logit of the median request at the level of the capacity.
    """
    gap = alpha - threshold
    if gap <= 0.0:
        return None
    zero = math.log2(REFERENCE_TOTAL_WORDS)
    start = float(plateau.value(0.0, plateau_params)[0])

    def logit(x: float) -> float:
        """Return the logit above the threshold at ``x = log₂ s`` of a text of ``s`` words."""
        level = float(plateau.value(max(x - zero, 0.0), plateau_params)[0])
        return gap + float(slice_value(slice_form, shared, level, np.array([x]))[0]) - start

    if logit(_X_ROOT_TOP) > 0.0:
        return None
    low = -_X_ROOT_TOP
    while logit(low) <= 0.0:
        if low < -_X_ROOT_FLOOR:
            return None
        low *= 2.0
    return float(brentq(logit, low, _X_ROOT_TOP, xtol=1e-12))


def capacity_by_bins(
    theta: Mapping[str, float],
    m_items: Sequence[AxisBin],
    r_items: Sequence[AxisBin],
    lambda_m: float,
    lambda_r: float,
    reserve: float,
) -> dict[str, Any]:
    """Return the capacity by the bins: the article, section 5.5.1.

    The jump of M between neighbouring bins is spread over the rules of the bin, R is the
    coefficient of the bin of the rule minus the shift ``λ_R``.

    Args:
        theta: The coefficients by name.
        m_items: The bins of the law of the number of rules.
        r_items: The bins of the law of the rule number.
        lambda_m: The shift ``λ_M`` of the law of the number of rules.
        lambda_r: The shift ``λ_R`` of the law of the rule number.
        reserve: The reserve of the level.

    Returns:
        ``value`` and ``status``: ``measured``; ``below_one_rule``; ``beyond_support`` with
        ``lower_bound`` and ``reserve_left`` - the reserve is not exhausted inside the bins.
    """
    if reserve <= 0.0:
        return {"value": None, "status": STATUS_BELOW_ONE_RULE}
    r_bins = {item.number: item for item in r_items}
    previous = -lambda_m
    price = 0.0
    for item in sorted(m_items, key=lambda entry: entry.number):
        corrected = float(theta[item.name]) - lambda_m
        rules = max(int(item.upper) - int(item.lower) + 1, 1)
        step = (corrected - previous) / rules
        previous = corrected
        r_item = r_bins.get(item.number)
        r_value = float(theta[r_item.name]) - lambda_r if r_item is not None else 0.0
        for rule in range(int(item.lower), int(item.upper) + 1):
            if rule < 2:
                continue
            cost = -step - r_value
            if price + cost >= reserve:
                return {"value": rule - 1 + (reserve - price) / cost, "status": STATUS_MEASURED}
            price += cost
    return {
        "value": None,
        "status": STATUS_BEYOND_SUPPORT,
        "lower_bound": int(max(item.upper for item in m_items)),
        "reserve_left": reserve - price,
    }


def _grid_law(
    design: VariableDesign,
    names: list[str],
    theta_vector: np.ndarray,
    covariance: np.ndarray,
    forms: Sequence[LawForm],
) -> _GridLaw | str:
    """Fit the slices of the grid by the form chosen among the candidates (``laws.grid_slices``).

    The weights are ``1/(σᵢ² + τ²)`` of the own variances of the cells, without their mutual
    dependence. The errors are still carried by the full covariance, through the sensitivity.

    Args:
        design: The folded matrix: the cells of the grid and their centres.
        names: The columns of the fit.
        theta_vector: The coefficients in the order of ``names``.
        covariance: Their covariance.
        forms: The candidate forms of the slices.

    Returns:
        The slices, or the reason of a grid whose slices no form describes.
    """
    cells = [cell for cell in design.cells if cell.name in names]
    cell_index = [names.index(cell.name) for cell in cells]
    x_cells = np.array([math.log2(design.abscissae[cell.name]["x"]) for cell in cells])
    slices = sorted({cell.v_bin for cell in cells})
    rows = np.array(
        [[x, float(slices.index(cell.v_bin))] for x, cell in zip(x_cells, cells, strict=True)]
    )
    cell_covariance = covariance[np.ix_(cell_index, cell_index)]
    law = choose_slice_form(
        forms, rows, theta_vector[cell_index], np.diag(cell_covariance), len(slices)
    )
    if isinstance(law, str):
        return law
    operator = law.fit.sensitivity[law.width :]
    level_variances = np.diag(operator @ cell_covariance @ operator.T)
    slice_u = []
    for v in slices:
        members = [cell for cell in cells if cell.v_bin == v]
        weight = np.array([cell.trials for cell in members], dtype=float)
        slice_u.append(
            float(
                (weight * [design.abscissae[cell.name]["u"] for cell in members]).sum()
                / weight.sum()
            )
        )
    return _GridLaw(
        cells=cells,
        cell_index=cell_index,
        slices=slices,
        law=law,
        level_variances=level_variances,
        slice_u=slice_u,
    )


def _plateau_law(grid: _GridLaw, size: int, forms: Sequence[LawForm]) -> BlockLaw | str:
    """Choose the law of the levels of the slices over ``u = log₂(v/20)``.

    The weights are ``1/(σᵢ² + τ²)`` of the own variances of the levels, as for the cells of the
    slices; the rows of the levels over the coefficients carry the errors.

    Args:
        grid: The slices of the grid.
        size: The number of the coefficients of the fit.
        forms: The candidate forms of the block.

    Returns:
        The law, or the reason of a block without a law.
    """
    count = len(grid.slices)
    if count < MIN_POINTS:
        return REASON_FEW_POINTS
    x = np.array(grid.slice_u)
    choice = choose_scattered_law(
        forms, x, grid.law.levels, np.maximum(grid.level_variances, 1e-24)
    )
    if isinstance(choice, str):
        return choice
    names = [f"v_bin_{v}" for v in grid.slices]
    return block_law(choice, grid.level_rows(size), names, x, at_infinity=False)


def _density_starts(values: np.ndarray, x: np.ndarray) -> list[np.ndarray]:
    """Return the starts of the density law: the level at the rarest bin, two slopes, three corners.

    Args:
        values: The coefficients of the bins.
        x: Their abscissae.
    """
    top = float(x.max())
    return [
        np.array([float(values[np.argmin(x)]), slope, share * top, 1.0])
        for slope in (-1.0, -3.0)
        for share in (0.3, 0.6, 0.8)
    ]


def _rule_starts(values: np.ndarray, x: np.ndarray) -> list[np.ndarray]:
    """Return the starts of the law of the rule number: the first bin, two slopes, three corners.

    Args:
        values: The coefficients of the bins.
        x: Their abscissae.
    """
    top = float(x.max())
    return [
        np.array([float(values[0]), slope, share * top, 1.0])
        for slope in (-0.5, -1.0)
        for share in (0.3, 0.5, 0.7)
    ]


def _axis_law(
    design: VariableDesign,
    fit: VariableFit,
    theta_vector: np.ndarray,
    covariance: np.ndarray,
    letter: str,
    forms: Sequence[LawForm],
    *,
    at_infinity: bool = False,
    starts: Callable[[np.ndarray, np.ndarray], list[np.ndarray]] | None = None,
) -> tuple[BlockLaw | str, list[AxisBin]]:
    """Choose the law of an axis over the binary logarithm of its load with the weights ``1/(σᵢ² + τ²)``.

    The abscissa is ``log₂(p/2⁻¹⁰)`` for the density, ``log₂ m`` and ``log₂ r`` for the rules.

    Args:
        design: The folded matrix: the table of the bins of the axis and their centres.
        fit: The fit of the quantity.
        theta_vector: The coefficients in the order of the fit.
        covariance: Their covariance.
        letter: The axis: ``P``, ``M`` or ``R``.
        forms: The candidate forms of the block.
        at_infinity: The zero of the axis lies at minus infinity of the abscissa (the density).
        starts: The starts of the slanted hyperbola from the coefficients and the abscissae.

    Returns:
        The law or the reason of a block without a law, and the bins of the law.
    """
    names = list(fit.names)
    items = _axis_items(design, fit, letter)
    if len(items) < MIN_POINTS:
        return REASON_FEW_POINTS, items
    index = [names.index(item.name) for item in items]
    key = letter.lower()
    centers = np.array([design.abscissae[item.name][key] for item in items])
    x = np.log2(centers / P_ZERO) if letter == "P" else np.log2(centers)
    values = theta_vector[index]
    own = own_variances(covariance[np.ix_(index, index)], implicit_reference=True)[1:]
    given = None if starts is None else {"slanted_hyperbolic": starts(values, x)}
    choice = choose_scattered_law(forms, x, values, own, starts=given)
    if isinstance(choice, str):
        return choice, items
    law = block_law(
        choice, unit_rows(index, len(names)), [item.name for item in items], x, at_infinity
    )
    return law, items


def _capacities(
    stack: LawStack,
    laws: Mapping[str, BlockLaw],
    slice_form: LawForm,
    level: Callable[[np.ndarray], float],
    joint: np.ndarray,
    theta: Mapping[str, float],
    m_items: Sequence[AxisBin],
    r_items: Sequence[AxisBin],
    support_words: float,
) -> list[dict[str, Any]]:
    """Return the capacities of H by the laws and the instruction capacity by the bins.

    A capacity is a function of the law parameters, and its half-width goes by the delta method: the
    central differences of the capacity by every parameter, carried through the rows of the
    parameters to the coefficients of the fit.

    Args:
        stack: The parameters of the laws and their rows.
        laws: The chosen laws by their parts of the stack.
        slice_form: The form of the slices of the grid.
        level: The level ``α`` from a vector of the stack.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        theta: The coefficients by name.
        m_items: The bins of the law of the number of rules.
        r_items: The bins of the law of the rule number.
        support_words: The longest text of the measurement, in words.

    Returns:
        The instruction capacity by the law and by the bins and the fatigue capacity.
    """
    table, rule, plateau = laws["table"], laws["rule"], laws["plateau"]

    def instruction(vector: np.ndarray) -> float:
        """Return ``log₂`` of the instruction capacity: the root of the cumulative price."""
        return log_capacity(
            table,
            stack.part(vector, "table"),
            rule,
            stack.part(vector, "rule"),
            level(vector) - LEVEL_THRESHOLD,
        )

    def fatigue_at(vector: np.ndarray, alpha: float) -> float:
        """Return ``log₂`` of a fatigue capacity at the level ``alpha``; ``nan`` - no root."""
        found = log_fatigue(
            slice_form,
            stack.part(vector, "slices"),
            plateau,
            stack.part(vector, "plateau"),
            alpha,
            LEVEL_THRESHOLD,
        )
        return math.nan if found is None else found

    def fatigue(vector: np.ndarray) -> float:
        """Return ``log₂`` of the fatigue capacity; ``nan`` - the law has no root."""
        return fatigue_at(vector, level(vector))


    def words_row(key: str, function: Callable[[np.ndarray], float]) -> dict[str, Any]:
        """Return the row of a capacity in words; a law without a root has no value."""
        found = function(stack.vector)
        if math.isnan(found):
            return {
                "key": key,
                "level": LEVEL,
                "method": "law",
                "value": None,
                "status": STATUS_NO_ROOT,
            }
        return capacity_row(
            key, LEVEL, found, half_width(stack.gradient(function), joint), support_words
        )

    alpha = level(stack.vector)
    support_rules = float(max(item.upper for item in m_items))
    lambda_m = table.shift()
    lambda_r = rule.shift()
    return [
        capacity_row(
            "instruction_capacity",
            LEVEL,
            instruction(stack.vector),
            half_width(stack.gradient(instruction), joint),
            support_rules,
        ),
        {
            "key": "instruction_capacity",
            "level": LEVEL,
            "method": "bins",
            **capacity_by_bins(
                theta, m_items, r_items, lambda_m, lambda_r, alpha - LEVEL_THRESHOLD
            ),
        },
        words_row("fatigue_capacity_words", fatigue),
    ]


def _slice_rows(grid: _GridLaw, plateau: BlockLaw) -> list[dict[str, Any]]:
    """Return the slices of the grid: their level with its half-width and the law of the levels.

    Args:
        grid: The slices of the grid.
        plateau: The law of the levels.
    """
    return [
        {
            "v_bin": v,
            "words": next(
                [cell.lower_words, cell.upper_words] for cell in grid.cells if cell.v_bin == v
            ),
            "u": grid.slice_u[slot],
            "mu_v": float(grid.law.levels[slot]),
            "half": HALF * math.sqrt(max(float(grid.level_variances[slot]), 0.0)),
            "law": float(plateau.value(grid.slice_u[slot])[0]),
        }
        for slot, v in enumerate(grid.slices)
    ]


def _rule_outcome(point: PointTrials) -> tuple[float, float]:
    """Return the trials of a point and its failures of H."""
    return float(sum(row[2] for row in point.rule)), float(sum(row[3] for row in point.rule))


def rule_word_fit(
    points: Sequence[PointTrials], layout: BinLayout, weights: Mapping[str, float]
) -> SectionFit:
    """Fit H over the trials of the answered points: the first stage of its section.

    Args:
        points: The trials of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.
    """
    design = build_design(points, "H", layout, weights)
    run_weight = np.array([weights[key] for key in design.point_keys], dtype=float)
    fit = checked_fit(
        design,
        run_weight,
        request_loads(points, layout),
        requests_of(points, layout, _rule_outcome),
        axis_bins(design, ("P", "M")),
    )
    return SectionFit(light_design(design), fit, float(max(point.D * point.n for point in points)))


def rule_word_from_fit(
    section: SectionFit, layout: BinLayout, assignment: LawAssignment
) -> dict[str, Any]:
    """Compute the section of H over its first stage: the laws, the level, the capacities."""
    del layout
    assert isinstance(section.design, VariableDesign) and section.support is not None
    return rule_word_laws(section.design, section.fit, section.support, assignment)


def rule_word_section(
    points: Sequence[PointTrials],
    layout: BinLayout,
    weights: Mapping[str, float],
    assignment: LawAssignment,
) -> dict[str, Any]:
    """Compute the section of H over the trials of the answered points.

    Args:
        points: The trials of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.
        assignment: The forms of the laws of the passport.

    Returns:
        The section: ``model``, ``weights``, ``fit``, ``status``, ``laws``, ``shifts``, ``alpha``,
        ``capacities``, ``bins``. Without a fit ``status`` is ``no_laws``.
    """
    return rule_word_from_fit(rule_word_fit(points, layout, weights), layout, assignment)


def rule_word_laws(
    design: VariableDesign, fit: VariableFit, support_words: float, assignment: LawAssignment
) -> dict[str, Any]:
    """Compute the section of H over a fit of model VS-PMR.

    Args:
        design: The folded matrix of the model.
        fit: Its fit.
        support_words: The longest text of the measurement, in words.
        assignment: The forms of the laws of the passport.

    Returns:
        The section, as :func:`rule_word_section`. A block without a law leaves the section without
        laws: ``status`` is ``no_laws`` and ``law_reasons`` names the block and the reason.
    """
    section: dict[str, Any] = {
        "model": "VS-PMR",
        "weights": "V x P x M",
        "fit": fit_summary(fit),
        "status": STATUS_NO_LAWS,
    }
    if fit.spread is not None:
        section["spread_laws"] = spread_laws(
            section["fit"]["spread"], spread_forms(assignment, "H")
        )
    if fit.method == METHOD_NONE:
        return section
    names = list(fit.names)
    theta_vector = np.asarray(fit.theta, dtype=float)
    theta = dict(zip(names, fit.theta, strict=True))
    joint = joint_covariance(fit)
    size = len(names)
    covariance = joint[:size, :size]

    grid = _grid_law(
        design, names, theta_vector, covariance, assignment.forms_of("H", "grid_slices")
    )
    if isinstance(grid, str):
        section["law_reasons"] = {"grid_slices": grid}
        return section
    plateau = _plateau_law(grid, size, assignment.forms_of("H", "grid_plateau"))
    table, m_items = _axis_law(
        design, fit, theta_vector, covariance, "M", assignment.forms_of("H", "table_size")
    )
    density, _ = _axis_law(
        design,
        fit,
        theta_vector,
        covariance,
        "P",
        assignment.forms_of("H", "density"),
        at_infinity=True,
        starts=_density_starts,
    )
    rule, r_items = _axis_law(
        design,
        fit,
        theta_vector,
        covariance,
        "R",
        assignment.forms_of("H", "rule_position"),
        starts=_rule_starts,
    )
    chosen = {
        "grid_plateau": plateau,
        "table_size": table,
        "density": density,
        "rule_position": rule,
    }
    reasons = {block: law for block, law in chosen.items() if isinstance(law, str)}
    if reasons:
        section["law_reasons"] = reasons
        return section
    assert isinstance(plateau, BlockLaw) and isinstance(table, BlockLaw)
    assert isinstance(density, BlockLaw) and isinstance(rule, BlockLaw)
    laws = {"plateau": plateau, "table": table, "density": density, "rule": rule}
    stack = LawStack(
        {
            "slices": (grid.law.shared, grid.shared_rows(size)),
            **{key: (law.params, law.rows) for key, law in laws.items()},
        }
    )
    part_names = {
        "slices": grid.law.shared_names,
        **{key: law.form.parameters for key, law in laws.items()},
    }

    def shift_of(key: str) -> Callable[[np.ndarray], float]:
        """Return the shift of a block as a function of a vector of the stack."""
        return lambda vector: laws[key].shift(stack.part(vector, key))

    def level(vector: np.ndarray) -> float:
        """Return the level ``α = μ₀ + μ_P + λ_M + λ_R`` from a vector of the stack."""
        return sum(shift_of(key)(vector) for key in laws)

    def density_loss(vector: np.ndarray) -> float:
        """Return the loss of the density law from ``p → 0`` to ``p = 1``: ``μ_P − P(1)``."""
        params = stack.part(vector, "density")
        return density.shift(params) - float(density.value(_U_AT_ONE, params)[0])

    shift_rows = {key: stack.gradient(shift_of(key)) for key in laws}
    alpha = level(stack.vector)
    loss = density_loss(stack.vector)
    capacities = _capacities(
        stack, laws, grid.law.form, level, joint, theta, m_items, r_items, support_words
    )
    section.update(
        {
            "status": STATUS_MEASURED,
            "laws": {
                "grid": slice_entry(grid.law, grid.shared_rows(size), joint, "ℓ"),
                "grid_plateau": law_entry(plateau, joint, _SYMBOLS["grid_plateau"]),
                "density": {
                    **law_entry(density, joint, _SYMBOLS["density"]),
                    "at_one": density.shift() - loss,
                },
                "table_size": law_entry(table, joint, _SYMBOLS["table_size"]),
                "rule_position": law_entry(rule, joint, _SYMBOLS["rule_position"]),
            },
            "shifts": {
                "mu_0": {"value": plateau.shift(), "half": half_width(shift_rows["plateau"], joint)},
                "mu_P": {"value": density.shift(), "half": half_width(shift_rows["density"], joint)},
                "lambda_M": {"value": table.shift(), "half": half_width(shift_rows["table"], joint)},
                "lambda_R": {"value": rule.shift(), "half": half_width(shift_rows["rule"], joint)},
            },
            "alpha": {"value": alpha, "half": half_width(stack.gradient(level), joint)},
            "law_covariance": stack.covariance_entry(part_names, joint),
            "capacities": capacities,
            "bins": {
                "grid": cell_rows(
                    design,
                    fit,
                    plateau.shift(),
                    shift_rows["plateau"],
                    joint,
                    lambda cell, point: float(
                        grid.law.curve(
                            float(grid.law.levels[grid.slices.index(cell.v_bin)]),
                            np.array([math.log2(point["x"])]),
                        )[0]
                    ),
                ),
                "slices": _slice_rows(grid, plateau),
                "P": axis_rows(
                    design,
                    fit,
                    "P",
                    "p",
                    list(density.kept_bins),
                    lambda p: float(density.value(math.log2(p / P_ZERO))[0]),
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
                "R": axis_rows(
                    design,
                    fit,
                    "R",
                    "r",
                    list(rule.kept_bins),
                    lambda r: float(rule.value(math.log2(r))[0]),
                    rule.shift(),
                    shift_rows["rule"],
                    joint,
                ),
            },
        }
    )
    return section
