"""The text of the section of F in the passport: the fit, the laws, the level, the figures.

The capacities of F stand in the part of the capacities at the start of the document (``capacities_text``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iec.passport.document.number_text import number
from iec.passport.document.text_cells import (
    figure_links,
    fit_line,
    law_sentence,
    spread_lines,
    with_half,
)

#: The title of the quantity in the document.
TITLE = "F: a misfire on a background word on its place"


def background_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of F.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = [f"## {TITLE}", "", fit_line(section), ""]
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
        "The laws run through the bins without the poorly determined ones; a bin weighs `1/(σᵢ² + τ²)`, "
        "its own variance and the scatter of the bins around the law, and the errors are carried by the "
        "full covariance of the bins.",
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
        *figure_links("F", figures),
    ]
