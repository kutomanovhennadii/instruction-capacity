"""The curves and the labels of the laws of the blocks, by the forms the data chose.

A figure draws a law with the form that the section chose for its block among the candidates of the
assignment of forms (``config/passport_laws.yaml``, ``laws.block_choice``) and with the parameters
that the section publishes: the figures have no formulas of their own. A view of a block finds the
chosen form by its name among the candidates of the block, maps the published parameters onto the
vector of the form, gives the form its abscissa - the logarithm of the load or the load itself - and
writes the label with the numbers of the law, as the text of the passport writes its numbers. The
registries give the view of a block by the pair of its quantity and its name in the assignment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.document.number_text import number, parameter
from iec.passport.laws.block_choice import form_parameters
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.laws.grid_slices import published_shared, slice_value

#: Error code of a block of the plan of the figures that no view draws.
FIGURE_VIEW_UNKNOWN = "FIGURE_VIEW_UNKNOWN"
#: The zero of the logarithm of the density.
P_ZERO = 2.0**-10
#: The shortest text of the bench, in words: the zero of the logarithm of the length.
REFERENCE_WORDS = 20.0
#: The names of the forms in the labels.
FORM_LABELS = {
    "linear": "line",
    "slanted_hyperbolic": "slanted hyperbola",
    "asymptotic_hyperbolic": "two-asymptote hyperbola",
    "two_slope_hyperbolic": "two-slope hyperbola",
    "binary_exponential": "binary exponential",
    "saturation": "saturation",
    "hyperbolic": "hyperbola",
    "exponential": "exponential",
    "logarithmic": "logarithmic law",
    "log_linear_load": "logarithm plus line",
}

Curve = Callable[[np.ndarray], np.ndarray]
ViewT = TypeVar("ViewT")


@dataclass(frozen=True, slots=True, eq=False)
class AxisView:
    """The bins of an axis and its law: what a figure of the axis draws.

    ``rows`` - the bins; ``curve`` - the law; ``label`` - its label; ``grid`` - the abscissae of the
    curve; ``shift`` - the shift of the axis; ``name`` - the axis in the title; ``xlabel`` - the label
    of the abscissa; ``log_x`` - a binary logarithmic abscissa.
    """

    rows: Sequence[Mapping[str, Any]]
    curve: Curve
    label: str
    grid: np.ndarray
    shift: float
    name: str
    xlabel: str
    log_x: bool = True


@dataclass(frozen=True, slots=True, eq=False)
class SliceView:
    """The law of every determined length slice of the grid, its label and the shift of the grid."""

    slices: Mapping[int, Curve]
    label: str
    shift: float


@dataclass(frozen=True, slots=True, eq=False)
class PlateauView:
    """The levels of the slices by the text length, the law of the level and its abscissae."""

    words: Sequence[float]
    levels: Sequence[float]
    halves: Sequence[float]
    curve: Curve
    label: str
    grid: np.ndarray


def chosen_form(law: Mapping[str, Any], forms: Sequence[LawForm]) -> tuple[LawForm, np.ndarray]:
    """Return the form a law chose among the candidates of its block and its parameters.

    The line taken when no candidate converged is the line of the collection.

    Args:
        law: The published law: ``law_form`` and ``params``.
        forms: The candidate forms of the block.
    """
    form = next((form for form in forms if form.name == law["law_form"]), LinearForm())
    return form, form_parameters(law, form)


def law_label(law: Mapping[str, Any], form: LawForm) -> str:
    """Return the label of a law: its form, its parameters and its ``χ²``."""
    params = ", ".join(f"{name} = {parameter(float(law['params'][name]))}" for name in form.parameters)
    return f"{FORM_LABELS.get(form.name, form.name)}: {params}; χ² = {number(float(law['chi2']), digits=1)}"


def axis_grid(rows: Sequence[Mapping[str, Any]], low: float | None = None) -> np.ndarray:
    """Return the abscissae of a law curve: from the zero of the axis or the first bin, if it is
    further left, to the last.

    Args:
        rows: The rows of the bins of the axis.
        low: The zero of the axis, or ``None``.
    """
    centers = [float(row["center"]) for row in rows if "raw" in row]
    return np.geomspace(min(centers) if low is None else min(low, *centers), max(centers), 300)


def _axis_view(
    section: Mapping[str, Any],
    forms: Sequence[LawForm],
    law_key: str,
    bins_key: str,
    abscissa: Callable[[np.ndarray], np.ndarray],
    shift: float,
    name: str,
    xlabel: str,
    low: float | None,
) -> AxisView:
    """Return the view of an axis whose law goes over ``abscissa`` of the load of its bins."""
    law = section["laws"][law_key]
    form, params = chosen_form(law, forms)
    rows = section["bins"][bins_key]
    return AxisView(
        rows=rows,
        curve=lambda load: form.value(params, abscissa(np.asarray(load, dtype=float))),
        label=law_label(law, form),
        grid=axis_grid(rows, low),
        shift=shift,
        name=name,
        xlabel=xlabel,
    )


def sync_density(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the density of P_sync: its law over ``log₂(p/2⁻¹⁰)``."""
    return _axis_view(
        section, forms, "density", "P", lambda p: np.log2(p / P_ZERO),
        float(section["shifts"]["delta_P"]["value"]), "replacement density",
        "p = m/D: replacement density (centre of mass of the bin)", None,
    )


def rule_density(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the density of H: its law over ``log₂(p/2⁻¹⁰)``."""
    return _axis_view(
        section, forms, "density", "P", lambda p: np.log2(p / P_ZERO),
        float(section["shifts"]["mu_P"]["value"]), "replacement density",
        "p = m/D: replacement density (centre of mass of the bin)", None,
    )


def sync_table(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the number of rules of P_sync: its law over ``log₂ m``."""
    return _axis_view(
        section, forms, "table_size", "M", np.log2, float(section["shifts"]["delta_M"]["value"]),
        "number of rules", "m: number of rules (centre of mass of the bin)", 1.0,
    )


def rule_table(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the number of rules of H: its law over ``log₂ m``."""
    return _axis_view(
        section, forms, "table_size", "M", np.log2, float(section["shifts"]["lambda_M"]["value"]),
        "number of rules", "m: number of rules (centre of mass of the bin)", 1.0,
    )


def rule_position(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the rule number of H: its law over ``log₂ r``."""
    return _axis_view(
        section, forms, "rule_position", "R", np.log2,
        float(section["shifts"]["lambda_R"]["value"]), "rule number",
        "r: rule number in the list (centre of mass of the bin)", 1.0,
    )


def background_length(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the length of F: its law over ``log₂(v/20)``."""
    return _axis_view(
        section, forms, "V", "V", lambda v: np.log2(v / REFERENCE_WORDS),
        float(section["laws"]["V"]["at_20"]), "text length",
        "v = D·n: text length, words (centre of mass of the bin)", REFERENCE_WORDS,
    )


def background_share(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the share of the answer of F: its law over ``q = s/v`` on a linear abscissa."""
    share = section["laws"]["q"]
    form, params = chosen_form(share, forms)
    return AxisView(
        rows=section["bins"]["q"],
        curve=lambda q: form.value(params, np.asarray(q, dtype=float)),
        label=f"{law_label(share, form)}; at q = 0 {number(float(share['at_0']))}",
        grid=np.linspace(0.05, 1.0, 100) if form.name in ("hyperbolic", "logarithmic") else np.linspace(0.0, 1.0, 100),
        shift=float(share["at_0"]),
        name="share of the answer",
        xlabel="q = s/v: share of the answer written",
        log_x=False,
    )


def background_density(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the density of F: its law over ``log₂(p/2⁻¹⁰)``, the top of its hill the shift."""
    density = section["laws"]["P"]
    view = _axis_view(
        section, forms, "P", "P", lambda p: np.log2(p / P_ZERO), float(density["at_top"]),
        "replacement density", "p = m/D: replacement density (centre of mass of the bin)", P_ZERO,
    )
    label = (
        f"{view.label}; the top at p = {number(float(density['p_top']), digits=4)} is "
        f"{number(float(density['at_top']))}"
    )
    return AxisView(view.rows, view.curve, label, view.grid, view.shift, view.name, view.xlabel)


def background_rules(section: Mapping[str, Any], forms: Sequence[LawForm]) -> AxisView:
    """Return the number of rules of F: its law over ``log₂ m``."""
    return _axis_view(
        section, forms, "M", "M", np.log2, float(section["shifts"]["M(1)"]["value"]),
        "number of rules", "m: number of rules (centre of mass of the bin)", 1.0,
    )


def _slices(
    section: Mapping[str, Any],
    forms: Sequence[LawForm],
    rows: Sequence[Mapping[str, Any]],
    level_of: Callable[[Mapping[str, Any]], float],
    shift: float,
) -> SliceView:
    """Return the slices of a grid by the form the law of the slices chose, every slice at its level."""
    law = section["laws"]["grid"]
    form = next(item for item in forms if item.name == law["law_form"])
    shared = published_shared(law, form)

    def slice_curve(level: float) -> Curve:
        """Return the curve of a slice of a level."""
        return lambda s: slice_value(form, shared, level, np.log2(s))

    params = ", ".join(f"{name} = {parameter(float(value), digits=3)}" for name, value in law["params"].items())
    return SliceView(
        slices={int(row["v_bin"]): slice_curve(level_of(row)) for row in rows},
        label=f"{FORM_LABELS.get(form.name, form.name)} of a slice: {params}, every slice its own level μ",
        shift=shift,
    )


def sync_slices(section: Mapping[str, Any], forms: Sequence[LawForm]) -> SliceView:
    """Return the determined slices of P_sync, every one at its level in the coding of the fit."""
    shift = float(section["shifts"]["delta_VS"]["value"])
    determined = [row for row in section["bins"]["slices"] if row.get("determined")]
    return _slices(section, forms, determined, lambda row: float(row["mu_v"]) + shift, shift)


def rule_slices(section: Mapping[str, Any], forms: Sequence[LawForm]) -> SliceView:
    """Return the slices of H, every one at its level."""
    return _slices(
        section,
        forms,
        section["bins"]["slices"],
        lambda row: float(row["mu_v"]),
        float(section["shifts"]["mu_0"]["value"]),
    )


def sync_plateau(section: Mapping[str, Any], forms: Sequence[LawForm]) -> PlateauView:
    """Return the plateaus of P_sync: the law of the plateau over ``log₂(v/20)``."""
    law = section["laws"]["grid_plateau"]
    form, params = chosen_form(law, forms)
    shift = float(section["shifts"]["delta_VS"]["value"])
    determined = [row for row in section["bins"]["slices"] if row.get("determined")]
    words = [REFERENCE_WORDS * 2.0 ** float(row["u"]) for row in determined]
    return PlateauView(
        words=words,
        levels=[float(row["mu_v"]) + shift for row in determined],
        halves=[1.96 * float(row["se_fit"]) for row in determined],
        curve=lambda v: form.value(params, np.log2(np.asarray(v, dtype=float) / REFERENCE_WORDS)),
        label=f"law of the plateau, {law_label(law, form)}",
        grid=np.geomspace(REFERENCE_WORDS, max(words) * 1.3, 300),
    )


def rule_plateau(section: Mapping[str, Any], forms: Sequence[LawForm]) -> PlateauView:
    """Return the levels of the slices of H: the law of the levels over ``log₂(v/20)``."""
    law = section["laws"]["grid_plateau"]
    form, params = chosen_form(law, forms)
    rows = section["bins"]["slices"]
    words = [REFERENCE_WORDS * 2.0 ** float(row["u"]) for row in rows]
    return PlateauView(
        words=words,
        levels=[float(row["mu_v"]) for row in rows],
        halves=[float(row["half"]) for row in rows],
        curve=lambda v: form.value(params, np.log2(np.asarray(v, dtype=float) / REFERENCE_WORDS)),
        label=f"law of the levels, {law_label(law, form)}",
        grid=np.geomspace(REFERENCE_WORDS, max(words) * 1.3, 300),
    )


#: The views of the axes by the pair «quantity, block» of the assignment of forms.
AXIS_VIEWS: Mapping[tuple[str, str], Callable[[Mapping[str, Any], Sequence[LawForm]], AxisView]] = {
    ("P_sync", "density"): sync_density,
    ("P_sync", "table_size"): sync_table,
    ("H", "density"): rule_density,
    ("H", "table_size"): rule_table,
    ("H", "rule_position"): rule_position,
    ("F", "length"): background_length,
    ("F", "answer_share"): background_share,
    ("F", "density"): background_density,
    ("F", "table_size"): background_rules,
}
#: The views of the slices of the grid by the pair «quantity, block».
SLICE_VIEWS: Mapping[tuple[str, str], Callable[[Mapping[str, Any], Sequence[LawForm]], SliceView]] = {
    ("P_sync", "grid_slices"): sync_slices,
    ("H", "grid_slices"): rule_slices,
}
#: The views of the plateaus of the slices by the pair «quantity, block».
PLATEAU_VIEWS: Mapping[
    tuple[str, str], Callable[[Mapping[str, Any], Sequence[LawForm]], PlateauView]
] = {
    ("P_sync", "grid_plateau"): sync_plateau,
    ("H", "grid_plateau"): rule_plateau,
}


def known_view(views: Mapping[tuple[str, str], ViewT], quantity: str, block: str) -> ViewT:
    """Return the view of a block from a registry.

    Args:
        views: The registry: ``AXIS_VIEWS``, ``SLICE_VIEWS`` or ``PLATEAU_VIEWS``.
        quantity: The quantity.
        block: The block in the assignment of forms.

    Raises:
        ContractViolationError: ``FIGURE_VIEW_UNKNOWN`` - the registry has no view of the block.
    """
    view = views.get((quantity, block))
    if view is None:
        raise ContractViolationError(
            f"no figure draws the block '{block}' of {quantity}", error_code=FIGURE_VIEW_UNKNOWN
        )
    return view
