"""The cells of the text of the passport: the fit, the capacities, the spread, the links to the figures.

The document computes nothing: every number comes from the passport dictionary. A capacity is written
by the law with its interval, a value beyond the measured area marked as a continuation of the law,
and next to it by the bins; what a status says is a table, not a chain of branches. The approximations
of the spread are written by the drawings of ``law_drawings``, which the figures draw too.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from iec.passport.document.law_drawings import SPREAD_SYMBOLS, spread_drawing
from iec.passport.document.number_text import number, size
from iec.passport.measurement_files import FIGURES_DIRNAME

#: The names of the capacities in the table.
CAPACITY_NAMES = {
    "fatigue_capacity_words": "fatigue capacity, words",
    "density_odds_factor": "how many times full density multiplies the odds of an error",
    "instruction_capacity": "instruction capacity, rules",
}
#: The methods of a fit in the sentence about it.
FIT_METHODS = {
    "glmm_heterogeneous": (
        "GLMM with heterogeneous random-intercept variance, that is with the shift of a request "
        "whose variance depends on the load"
    ),
    "glmm": "with the shift of a request",
    "logit": (
        "the logit without the shift of a request: the shift did not converge, and the numbers are "
        "not comparable with other measurements"
    ),
    "none": "the fit did not converge",
}
#: The names of the parameters of the grid that its fit may hold on a bound of its search.
HELD_NAMES = {"gamma": "exponent γ"}
#: The names of the forms of the laws in the text.
FORM_NAMES = {
    "linear": "a line",
    "slanted_hyperbolic": "a slanted hyperbola",
    "two_slope_hyperbolic": "a two-slope hyperbola",
    "binary_exponential": "a binary exponential",
    "saturation": "a saturation",
    "hyperbolic": "a hyperbola",
    "exponential": "an exponential",
    "logarithmic": "a logarithmic law",
    "log_linear_load": "a logarithm plus a line",
}
#: The axes of the spread and their names in the table.
SPREAD_AXES = (("V", "text length, words"), ("P", "density"), ("M", "number of rules"))
#: The mark of a capacity by a law by its status; a status without a mark is written plain.
LAW_MARKS = {
    "extrapolated": ", continuation of the law",
    "conditional": ", conditional: a continuation of the law of the density",
}
#: The text of a capacity by the bins that has no value, by its status.
BINS_TEXTS: Mapping[str, Callable[[Mapping[str, Any]], str]] = {
    "beyond_support": lambda row: (
        f"at least {row['lower_bound']}, reserve left {number(float(row['reserve_left']))}"
    ),
    "below_one_rule": lambda row: "below one rule",
}


def with_half(row: Mapping[str, Any]) -> str:
    """Write a quantity with the half-width of its interval.

    Args:
        row: ``value`` and ``half``.
    """
    return f"{number(float(row['value']))} ± {number(float(row['half']))}"


def on_bound(law: Mapping[str, Any]) -> str:
    """Name the parameters of a law that lie on a bound of the search.

    Args:
        law: A law with ``held``.
    """
    held = ", ".join(HELD_NAMES[name] for name in law["held"])
    return f"; on the bound of the search: {held}" if held else ""


def law_sentence(title: str, law: Mapping[str, Any]) -> str:
    """Write the law of a block: the form the data chose, its formula, parameters and fit.

    Args:
        title: The block as the sentence names it, such as ``The density``.
        law: The published law of the block (``laws.block_choice.law_entry``).
    """
    params = ", ".join(
        f"`{name} = {number(float(value))} ± {number(float(law['half'][name]))}`"
        for name, value in law["params"].items()
    )
    held = ", ".join(law["held"])
    bound = f"; on the bound of the search: {held}" if held else ""
    chosen = f"chosen by `χ² + 2k` among {len(law['candidates'])} forms"
    if law.get("fallback"):
        chosen = "the line: no candidate converged"
    form = FORM_NAMES.get(law["law_form"], law["law_form"])
    return (
        f"{title} is {form}, `{law['formula']}`: {params}; `χ² = {number(float(law['chi2']), 1)}` "
        f"over {law['n_points']} bins{bound}; {chosen}."
    )


def law_cell(row: Mapping[str, Any]) -> str:
    """Write a capacity by the law: the value, the interval, the mark of its status.

    Args:
        row: The row of the capacity by the law.
    """
    if row.get("value") is None:
        return "none: the law gives no root"
    low, high = row["interval"]
    mark = LAW_MARKS.get(row["status"], "")
    return f"{size(float(row['value']))} ({size(float(low))}–{size(float(high))}){mark}"


def bins_cell(row: Mapping[str, Any] | None) -> str:
    """Write a capacity by the bins: its value or the text of its status.

    Args:
        row: The row of the capacity by the bins, or ``None`` without one.
    """
    if row is None:
        return "-"
    text = BINS_TEXTS.get(row["status"])
    if text is not None:
        return text(row)
    return size(float(row["value"]))


def capacity_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return the table of the capacities of a quantity: capacity, level, law, bins.

    Args:
        rows: The capacities of the section.
    """
    lines = [
        "| Capacity | q | By the law | By the bins |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if row["method"] != "law":
            continue
        twin = next(
            (
                other
                for other in rows
                if other["method"] == "bins"
                and other["key"] == row["key"]
                and other["level"] == row["level"]
            ),
            None,
        )
        level = "-" if row.get("level") is None else f"{float(row['level']):g}"
        lines.append(
            f"| {CAPACITY_NAMES[row['key']]} | {level} | {law_cell(row)} | {bins_cell(twin)} |"
        )
    return lines


def fit_line(section: Mapping[str, Any]) -> str:
    """Return the sentence about the fit of a quantity.

    Args:
        section: The section of the quantity.
    """
    fit = section["fit"]
    sigma = "none" if fit["sigma2"] is None else number(float(fit["sigma2"]))
    return (
        f"Model {section['model']}, request weights by the cube "
        f"{section['weights'].replace(' x ', ' × ')}. Fit: {FIT_METHODS[fit['method']]}; the variance of "
        f"the request shift `σ² = {sigma}`; {fit['points']:,} requests, {fit['trials']:,} trials, "
        f"{fit['failures']:,} failures, {fit['parameters']} parameters; removed bins: "
        f"{', '.join(fit['dropped']) or 'none'}."
    )


def figure_links(folder: str, names: Sequence[str]) -> list[str]:
    """Return the figures of a quantity, one paragraph each.

    Args:
        folder: The folder of the quantity under ``figures/``.
        names: The names of the figures without the extension.
    """
    lines: list[str] = []
    for name in names:
        lines += [f"![{folder}: {name.replace('-', ' ')}]({FIGURES_DIRNAME}/{folder}/{name}.png)", ""]
    return lines


def spread_lines(section: Mapping[str, Any]) -> list[str]:
    """Return the table of the variance of the shift by the groups of the axes; no spread - nothing.

    Args:
        section: The section of the quantity.
    """
    spread = section["fit"].get("spread")
    if spread is None:
        return []
    lines = [
        f"The variance of the shift depends on the load: `ln σ_j = z_j·γ`, {spread['parameters']} "
        f"parameters more than one variance, `2·Δ ln L = {number(float(spread['loglik_gain']), 1)}`. "
        "`σ²` above is the mean over the requests. A group is taken with the other two axes averaged "
        "over the requests.",
        "",
        "| Axis | Centre of the group | Share of the requests | σ² |",
        "|---|---|---|---|",
    ]
    for letter, name in SPREAD_AXES:
        for row in spread[letter]:
            lines.append(
                f"| {name} | {float(row['center']):.4g} | {number(float(row['share']))} | "
                f"{number(float(row['sigma2']))} ± {number(float(row['half']))} |"
            )
    laws = section.get("spread_laws")
    if laws:
        lines += [
            "",
            "The approximations through the groups, by the residuals in `ln σ²`, serve the eye.",
            "",
            "| Axis | Approximation | Parameters | χ² |",
            "|---|---|---|---|",
        ]
        for letter, name in SPREAD_AXES:
            law = laws.get(letter)
            if law is None:
                continue
            drawing = spread_drawing(law)
            symbol = SPREAD_SYMBOLS[letter]
            lines.append(
                f"| {name} | `{drawing.formula(symbol)}` | {drawing.parameter_text(law, symbol)} | "
                f"{number(float(law['chi2']), 1)} over {law['n_points']} groups |"
            )
    return [*lines, ""]
