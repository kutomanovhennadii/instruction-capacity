"""The text of the section of H in the passport: the fit, the laws, the level, the figures.

The capacities of H stand in the part of the capacities at the start of the document (``capacities_text``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iec.passport.document.text_cells import (
    figure_links,
    fit_line,
    law_sentence,
    slice_sentence,
    spread_lines,
    with_half,
)

#: The title of the quantity in the document.
TITLE = "H: a rule word on its place is replaced correctly"


def rule_word_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of H.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = [f"## {TITLE}", "", fit_line(section), ""]
    if section["status"] != "measured":
        return [*lines, "There are no laws: the fit did not converge.", ""]
    laws = section["laws"]
    return [
        *lines,
        *spread_lines(section),
        f"{slice_sentence(laws['grid'])} The law of the levels runs through the levels of the slices.",
        "",
        law_sentence("The law of the levels of the slices", laws["grid_plateau"]),
        "",
        law_sentence("The law of the density", laws["density"]),
        "",
        law_sentence("The law of the number of rules", laws["table_size"]),
        "",
        law_sentence("The law of the rule number", laws["rule_position"]),
        "",
        f"The level `α = μ₀ + μ_P + λ_M + λ_R = {with_half(section['alpha'])}`.",
        "",
        *figure_links("H", figures),
    ]
