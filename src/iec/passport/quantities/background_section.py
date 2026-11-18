"""The section of F in the passport: model V-(s/v)-PM, laws, level, three capacities.

A trial is a background word standing on its place, a success - no misfire: no right-hand word of a
rule stands on the place of the word. The model is ``η = V(v) + q(s/v) + P(p) + M(m) + b``: bins of
the text length that carry the level, ten equal bins of the share of the answer ``q = s/v``, the
density and the number of rules (``background_design``).

The laws V, q, P and M go through all the bins of the axis except the poorly determined ones, the
effective number of requests with failures ``(Σw)²/Σw²`` below ten, a bin weighing ``1/(σᵢ² + τ²)``;
errors are by the full covariance of the fit. The forms are chosen by the data among the candidates
of the assignment of forms (``config/passport_laws.yaml``, ``laws.block_choice``): over
``log₂(v/20)``, ``log₂(p/2⁻¹⁰)`` and ``log₂ m`` the line, the slanted hyperbola and the binary
exponential, and for the density also the logarithmic law with a line ``a + b·x + c·p``, which
describes a hill at rare replacements; over ``q`` the exponential and the line; by the smallest
``QAIC``.

The section goes by the steps of every section: the facts of the fit and the approximations of the
spread, the laws of the blocks, the level, the capacities by the laws and by the bins, the rows of
the bins. The shift of an axis is its law at its zero: ``V(20)``, ``q(0)``, ``M(1)``, and for the
density ``δ_P``, the maximum of its law on ``[2⁻¹⁰, 1]``: the law can rise at rare replacements and
have a hill. The level is ``α = V(20) + q(0) + δ_P + M(1)``. Three capacities by the laws at the
level 0.5, where the threshold of the logit of the median request is zero at any variance of the
shift, so the reserve is the level itself: the instruction capacity, the number of rules at which
``α + [M(m) − M(1)]`` reaches zero, also by the bins (the article, section 5.5.1); the fatigue
capacity - the word number at the last word of a text of the same length, ``q = 1``, at which
``α − [q(0) − q(1)] + [V(s) − V(20)]`` reaches zero. Instead of the guaranteed fatigue capacity the
section gives how many times full density multiplies the odds of an error, ``e^(−P̂(1))``,
``P̂(1) = P(1) − P(p → 0)`` (decision 7 of the passport standard). There are no background words at
``p = 1``, so this measure is conditional: it always continues the law of the density, and it allows
comparing the effect of the density on P_sync, H and F.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from iec.passport.fitting.chain.fit_paths import METHOD_NONE
from iec.passport.fitting.load_spread import request_loads
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.block_choice import (
    MIN_POINTS,
    REASON_FEW_POINTS,
    BlockLaw,
    LawStack,
    block_law,
    block_shift,
    choose_scattered_law,
    law_entry,
    unit_rows,
)
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import STATUS_MEASURED
from iec.passport.laws.law_rows import (
    HALF,
    LEVEL,
    LEVEL_THRESHOLD,
    fit_summary,
    half_width,
    joint_covariance,
    own_variances,
)
from iec.passport.laws.spread_laws import spread_forms, spread_laws
from iec.passport.quantities.steps.median_check import AxisBins, checked_fit, requests_of
from iec.passport.quantities.background_design import (
    AnswerShareDesign,
    BackgroundCells,
    answer_share_design,
)
from iec.passport.quantities.steps.bins_walk import capacity_by_bin_walk
from iec.passport.quantities.steps.law_capacity import odds_factor_row, root_capacity_row
from iec.passport.trials.layout.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout

#: The zero of the logarithm of the density: the smallest density measured on the models.
P_ZERO = 2.0**-10
#: ``log₂(1/2⁻¹⁰)``: the density ``p = 1`` on the abscissa of its law.
_U_AT_ONE = 10.0
#: The grid of the search of the top of the density law over ``log₂(p/2⁻¹⁰)``.
_TOP_GRID = np.linspace(0.0, _U_AT_ONE, 2001)
#: The end of the search of the root of a capacity in binary logarithms.
_LOG_ROOT_TOP = 1000.0
#: The starts of the exponential law of the share of the answer.
_SHARE_STARTS = (
    np.array([0.0, 1.0, 0.4]),
    np.array([-0.2, 1.0, 1.0]),
    np.array([0.0, -1.0, 0.4]),
    np.array([-0.2, -1.0, 1.0]),
)
#: The symbols of the abscissae of the laws in their formulas.
_SYMBOLS = {"V": "log₂(v/20)", "q": "q", "P": "log₂(p/2⁻¹⁰)", "M": "log₂ m"}
STATUS_NO_LAWS = "no_laws"


def _good_bins(built: AnswerShareDesign, letter: str, names: Sequence[str]) -> list[str]:
    """Return the bins of the law of an axis in the order of the axis: those with a coefficient.

    The reference bin has no coefficient and no error: in the weighted fit it carries no weight, so
    it does not enter the law, as the reference bins of P_sync and H.

    Args:
        built: The folded matrix with the tallies of its bins.
        letter: The axis: ``V``, ``q``, ``P`` or ``M``.
        names: The columns of the fit.
    """
    reference = built.references.get(letter)
    good = [name for name in built.bins[letter] if name != reference and name in names]
    return sorted(good, key=lambda name: built.centers[letter][name])


def _abscissa(letter: str, centers: np.ndarray) -> np.ndarray:
    """Return the abscissae of the law of an axis from the centres of its bins."""
    if letter == "V":
        return np.log2(centers / REFERENCE_TOTAL_WORDS)
    if letter == "P":
        return np.log2(centers / P_ZERO)
    if letter == "M":
        return np.log2(centers)
    return centers


def _axis_law(
    built: AnswerShareDesign,
    letter: str,
    names: Sequence[str],
    theta: np.ndarray,
    covariance: np.ndarray,
    forms: Sequence[LawForm],
) -> BlockLaw | str:
    """Choose the law of an axis through its bins with a coefficient.

    The weights are ``1/(σᵢ² + τ²)`` of the own variances of the bins, as in the other sections; the
    outliers leave the law by the clipping of ``block_choice``.

    Args:
        built: The folded matrix with the tallies of its bins.
        letter: The axis: ``V``, ``q``, ``P`` or ``M``.
        names: The columns of the fit.
        theta: The coefficients in the order of ``names``.
        covariance: Their covariance.
        forms: The candidate forms of the block.

    Returns:
        The law, or the reason of a block without a law.
    """
    good = _good_bins(built, letter, names)
    if len(good) < MIN_POINTS:
        return REASON_FEW_POINTS
    columns = [names.index(name) for name in good]
    values = np.array([float(theta[column]) for column in columns])
    x = _abscissa(letter, np.array([built.centers[letter][name] for name in good]))
    starts = {"exponential": list(_SHARE_STARTS)} if letter == "q" else None
    reference = built.references.get(letter)
    own = own_variances(
        covariance[np.ix_(columns, columns)], implicit_reference=reference is not None
    )
    choice = choose_scattered_law(
        forms, x, values, own[1 if reference is not None else 0 :], starts=starts
    )
    if isinstance(choice, str):
        return choice
    return block_law(choice, unit_rows(columns, len(names)), good, x, at_infinity=False)


def _law_at(law: BlockLaw, letter: str) -> Callable[[float], float]:
    """Return the law of an axis at the centre of a bin, in the load of the axis."""
    return lambda center: float(law.value(_abscissa(letter, np.array([center])))[0])


def _density_top(law: BlockLaw) -> float:
    """Return the abscissa of the maximum of the density law on ``[2⁻¹⁰, 1]``.

    The law can rise at rare replacements and have a hill; otherwise the maximum stands at an end of
    the range. The grid finds the hill and a bounded search between its neighbours refines the top.
    The top moves with the law, but the law is flat there, so the error of the shift is the error of
    the law at this abscissa.
    """
    values = law.value(_TOP_GRID)
    k = int(np.argmax(values))
    if k in (0, len(_TOP_GRID) - 1):
        return float(_TOP_GRID[k])
    found = minimize_scalar(
        lambda x: -float(law.value(x)[0]),
        bounds=(float(_TOP_GRID[k - 1]), float(_TOP_GRID[k + 1])),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(found.x)


def _root(function: Callable[[float], float]) -> float:
    """Return the root of a falling logit above the threshold on ``[0, _LOG_ROOT_TOP]``.

    A logit already at the threshold at zero gives zero; a logit that does not reach it gives
    ``nan`` - the law has no root.
    """
    if function(0.0) <= 0.0:
        return 0.0
    if function(_LOG_ROOT_TOP) > 0.0:
        return math.nan
    return float(brentq(function, 0.0, _LOG_ROOT_TOP, xtol=1e-12))


def _background_axis_rows(
    built: AnswerShareDesign,
    fit: VariableFit,
    letter: str,
    good: Sequence[str],
    law: Callable[[float], float],
    shift: float,
    shift_gain: np.ndarray,
    joint: np.ndarray,
) -> list[dict[str, Any]]:
    """Return the bins of an axis: raw, minus the shift, own errors, the law, the counts.

    Args:
        built: The folded matrix with the tallies of its bins.
        fit: The fit of the quantity.
        letter: The axis: ``V``, ``q``, ``P`` or ``M``.
        good: The bins of the law.
        law: The law of the axis at a centre.
        shift: The shift of the axis.
        shift_gain: The weights of the shift over the coefficients.
        joint: The joint covariance of the coefficients and of the variance of the shift.
    """
    names = list(fit.names)
    theta = np.asarray(fit.theta, dtype=float)
    bins = built.bins[letter]
    reference = built.references.get(letter)
    ordered = sorted(bins, key=lambda name: built.centers[letter][name])
    present = [name for name in ordered if name in names]
    index = [names.index(name) for name in present]
    own = own_variances(joint[np.ix_(index, index)], implicit_reference=reference is not None)
    offset = 1 if reference is not None else 0
    rows: list[dict[str, Any]] = []
    for name in ordered:
        center = built.centers[letter][name]
        row: dict[str, Any] = {
            "name": name,
            "center": center,
            "trials": bins[name].trials,
            "failures": bins[name].failures,
            "requests": bins[name].requests,
            "effective": bins[name].effective,
        }
        gradient = -shift_gain.copy()
        if name == reference:
            raw, q = 0.0, own[0]
        elif name in names:
            gradient[names.index(name)] += 1.0
            raw, q = float(theta[names.index(name)]), own[present.index(name) + offset]
        else:
            rows.append({**row, "status": "excluded"})
            continue
        if name == reference:
            status = "reference"
        else:
            status = "law" if name in good else "outlier"
        rows.append(
            {
                **row,
                "status": status,
                "in_law": name in good,
                "raw": raw,
                "own_half": HALF * math.sqrt(q),
                "restored": raw - shift,
                "restored_half": half_width(gradient, joint),
                "law": float(law(center)),
            }
        )
    return rows


def background_section(
    points: Sequence[BackgroundCells],
    layout: BinLayout,
    weights: Mapping[str, float],
    assignment: LawAssignment,
) -> dict[str, Any]:
    """Compute the section of F over the trials of the answered points.

    Args:
        points: The trials of F of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.
        assignment: The forms of the laws of the passport.

    Returns:
        The section: ``model``, ``weights``, ``fit``, ``status``, ``laws``, ``shifts``, ``alpha``,
        ``capacities``, ``bins``. Without a fit ``status`` is ``no_laws``.
    """
    built = answer_share_design(points, layout, weights)
    run_weight = np.array([weights[key] for key in built.design.point_keys], dtype=float)
    fit = checked_fit(
        built.design,
        run_weight,
        request_loads(points, layout),
        requests_of(points, layout, _background_outcome),
        _background_axes(built),
    )
    return background_laws(built, fit, assignment)


def _background_outcome(point: BackgroundCells) -> tuple[float, float]:
    """Return the trials of a point and its misfires of F."""
    return (
        float(sum(cell[1] for cell in point.cells)),
        float(sum(cell[2] for cell in point.cells)),
    )


def _background_axes(built: AnswerShareDesign) -> dict[str, AxisBins]:
    """Return the bins of the axes of F constant within a request: length, density, rules.

    The number of a bin is the number after the last ``_`` of its name; an axis without a reference
    bin is not checked.
    """
    axes: dict[str, AxisBins] = {}
    for letter in ("V", "P", "M"):
        reference = built.references.get(letter)
        if reference is None:
            continue
        axes[letter] = AxisBins(
            int(reference.rsplit("_", 1)[1]),
            {int(name.rsplit("_", 1)[1]): name for name in built.bins[letter]},
        )
    return axes


def background_laws(
    built: AnswerShareDesign, fit: VariableFit, assignment: LawAssignment
) -> dict[str, Any]:
    """Compute the section of F over a fit of model V-(s/v)-PM.

    Args:
        built: The folded matrix of the model with the tallies of its bins.
        fit: Its fit.
        assignment: The forms of the laws of the passport.

    Returns:
        The section, as :func:`background_section`. A block without a law leaves the section without
        laws: ``law_reasons`` names the block and the reason.
    """
    section: dict[str, Any] = {
        "model": "V-(s/v)-PM",
        "weights": "V x P x M",
        "fit": fit_summary(fit),
        "status": STATUS_NO_LAWS,
    }
    if fit.spread is not None:
        section["spread_laws"] = spread_laws(section["fit"]["spread"], spread_forms(assignment, "F"))
    if fit.method == METHOD_NONE:
        return section
    names = list(fit.names)
    theta = np.asarray(fit.theta, dtype=float)
    joint = joint_covariance(fit)
    blocks = {"V": "length", "q": "answer_share", "P": "density", "M": "table_size"}
    covariance = joint[: len(names), : len(names)]
    chosen = {
        letter: _axis_law(built, letter, names, theta, covariance, assignment.forms_of("F", block))
        for letter, block in blocks.items()
    }
    reasons = {blocks[letter]: law for letter, law in chosen.items() if isinstance(law, str)}
    if reasons:
        section["law_reasons"] = reasons
        return section
    laws: dict[str, BlockLaw] = {
        letter: law for letter, law in chosen.items() if isinstance(law, BlockLaw)
    }
    stack = LawStack({letter: (law.params, law.rows) for letter, law in laws.items()})
    top = _density_top(laws["P"])

    def part(vector: np.ndarray, letter: str) -> np.ndarray:
        """Return the parameters of the law of an axis from a vector of the stack."""
        return stack.part(vector, letter)

    def at(vector: np.ndarray, letter: str, x: float) -> float:
        """Return the law of an axis at its abscissa ``x``."""
        return float(laws[letter].value(x, part(vector, letter))[0])

    def shift_of(letter: str) -> Callable[[np.ndarray], float]:
        """Return the shift of an axis as a function of a vector of the stack."""
        if letter == "P":
            return lambda vector: at(vector, "P", top)
        return lambda vector: laws[letter].shift(part(vector, letter))

    def level(vector: np.ndarray) -> float:
        """Return the level ``α = V(20) + q(0) + δ_P + M(1)`` from a vector of the stack."""
        return sum(shift_of(letter)(vector) for letter in laws)

    def fatigue_level(vector: np.ndarray) -> float:
        """Return the level at the last word of a text, ``q = 1``."""
        return level(vector) - shift_of("q")(vector) + at(vector, "q", 1.0)

    def length_root(vector: np.ndarray, start: float) -> float:
        """Return ``log₂(s/20)`` at which ``start + [V(s) − V(20)]`` reaches the threshold."""
        zero = shift_of("V")(vector)
        return _root(lambda x: start + at(vector, "V", x) - zero - LEVEL_THRESHOLD)

    def instruction(vector: np.ndarray) -> float:
        """Return ``log₂`` of the instruction capacity by the law of the number of rules."""
        alpha, zero = level(vector), shift_of("M")(vector)
        return _root(lambda x: alpha + at(vector, "M", x) - zero - LEVEL_THRESHOLD)

    def fatigue(vector: np.ndarray) -> float:
        """Return ``log₂(s/20)`` of the fatigue capacity."""
        return length_root(vector, fatigue_level(vector))

    def density_change(vector: np.ndarray) -> float:
        """Return ``P̂(1)``: the change of the density law from ``p → 0`` to ``p = 1``."""
        params = part(vector, "P")
        zero = block_shift(laws["P"].form, params, True)
        return at(vector, "P", _U_AT_ONE) - zero

    shift_rows = {letter: stack.gradient(shift_of(letter)) for letter in laws}
    shifts = {letter: shift_of(letter)(stack.vector) for letter in laws}
    alpha = level(stack.vector)
    reference_m = built.references.get("M")
    m_rows = sorted(
        ([(built.centers["M"][reference_m], 0.0)] if reference_m is not None else [])
        + [
            (built.centers["M"][name], float(theta[names.index(name)]))
            for name in laws["M"].kept_bins
        ]
    )
    edge_rules = 2 ** math.ceil(math.log2(m_rows[-1][0]))
    words = float(REFERENCE_TOTAL_WORDS)
    capacities: list[dict[str, Any]] = [
        root_capacity_row("instruction_capacity", instruction, stack, joint, float(edge_rules)),
        {
            "key": "instruction_capacity",
            "level": LEVEL,
            "method": "bins",
            **capacity_by_bin_walk(m_rows, shifts["M"], alpha - LEVEL_THRESHOLD),
        },
        root_capacity_row(
            "fatigue_capacity_words",
            fatigue,
            stack,
            joint,
            float(built.largest_words),
            scale=words,
        ),
        # At p = 1 there are no background words: the measure continues the law of the density.
        odds_factor_row(density_change, stack, joint, conditional=True),
    ]
    q_law, p_law = laws["q"], laws["P"]
    section.update(
        {
            "status": STATUS_MEASURED,
            "laws": {
                "V": {
                    **law_entry(laws["V"], joint, _SYMBOLS["V"]),
                    "at_20": shifts["V"],
                },
                "q": {
                    **law_entry(q_law, joint, _SYMBOLS["q"]),
                    "at_0": shifts["q"],
                    "at_1": float(q_law.value(1.0)[0]),
                },
                "P": {
                    **law_entry(p_law, joint, _SYMBOLS["P"]),
                    "p_top": P_ZERO * 2.0**top,
                    "at_top": shifts["P"],
                    "at_one": float(p_law.value(_U_AT_ONE)[0]),
                },
                "M": law_entry(laws["M"], joint, _SYMBOLS["M"]),
            },
            "shifts": {
                "V(20)": {"value": shifts["V"], "half": half_width(shift_rows["V"], joint)},
                "q(0)": {"value": shifts["q"], "half": half_width(shift_rows["q"], joint)},
                "delta_P": {"value": shifts["P"], "half": half_width(shift_rows["P"], joint)},
                "M(1)": {"value": shifts["M"], "half": half_width(shift_rows["M"], joint)},
            },
            "alpha": {"value": alpha, "half": half_width(stack.gradient(level), joint)},
            "edge_rules": edge_rules,
            "edge_words": built.largest_words,
            "capacities": capacities,
            "bins": {
                letter: _background_axis_rows(
                    built,
                    fit,
                    letter,
                    list(laws[letter].kept_bins),
                    _law_at(laws[letter], letter),
                    shifts[letter],
                    shift_rows[letter],
                    joint,
                )
                for letter in ("V", "q", "P", "M")
            },
        }
    )
    return section
