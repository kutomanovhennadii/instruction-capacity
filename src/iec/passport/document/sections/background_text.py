"""The text of the section of F in the passport: the fit, the laws, the level, the capacities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iec.passport.document.number_text import number
from iec.passport.document.text_cells import (
    capacity_rows,
    figure_links,
    fit_line,
    law_sentence,
    spread_lines,
    with_half,
)


def background_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of F.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = ["## F: a misfire on a background word on its place", "", fit_line(section), ""]
    if section["status"] != "measured":
        return [*lines, "There are no laws: the fit did not converge.", ""]
    laws = section["laws"]
    shifts = ", ".join(
        f"`{name.replace('delta_P', 'δ_P')}` = {with_half(row)}"
        for name, row in section["shifts"].items()
    )
    return [
        *lines,
        *spread_lines(section),
        "The laws run through the bins without the poorly determined ones, by ordinary least squares; "
        "the form of a law is chosen by `χ² + 2k`, the residuals weighed by the covariance of the bins.",
        "",
        law_sentence("The law of the length", laws["V"]),
        "",
        law_sentence("The law of the share of the answer", laws["q"])
        + f" A fall of {number(laws['q']['at_0'] - laws['q']['at_1'])} over the whole answer.",
        "",
        law_sentence("The law of the density", laws["P"]),
        "",
        law_sentence("The law of the number of rules", laws["M"]),
        "",
        f"The level `α_F = V(20) + q(0) + δ_P + M(1) = {with_half(section['alpha'])}`, where `δ_P` is the "
        f"maximum of the law of the density, at `p = {number(float(laws['P']['p_top']), 4)}`; {shifts}. "
        f"The edge of the measured area: {section['edge_rules']} rules and {section['edge_words']:,} words.",
        "",
        *capacity_rows(section["capacities"]),
        "",
        "The instruction capacity is the number of rules at the zero point. The fatigue capacity is the "
        "number of the last word of a text of the same length at the zero point. Full density "
        "multiplies the odds of a misfire by `e^(−P̂(1))`, `P̂(1) = P(1) − P(p → 0)`; a factor below "
        "one would mean that full density lowers them. It is a "
        "conditional quantity: a text has no background words at `p = 1`, so it always continues the "
        "law of the density, and it allows comparing the effect of the density on `P_sync`, `H` and `F`.",
        "",
        *figure_links("F", figures),
    ]
