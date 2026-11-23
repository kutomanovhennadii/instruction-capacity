"""The text of the section of P_sync in the passport: the fit, the laws, the level, the figures.

The capacities of P_sync stand in the part of the capacities at the start of the document (``capacities_text``).
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
TITLE = "P_sync: the answer word stands against its reference word"


def sync_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of P_sync.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = [f"## {TITLE}", "", fit_line(section), ""]
    if section["status"] != "measured":
        return [*lines, "There are no laws: the fit or a law did not converge.", ""]
    laws = section["laws"]
    return [
        *lines,
        *spread_lines(section),
        f"{slice_sentence(laws['grid'])} The law of the plateau runs through the levels of the slices.",
        "",
        law_sentence("The law of the plateau", laws["grid_plateau"]),
        "",
        law_sentence("The law of the density", laws["density"]),
        "",
        law_sentence("The law of the number of rules", laws["table_size"]),
        "",
        f"The level `α = δ_VS + δ_P + δ_M = {with_half(section['alpha'])}` is the logit at the "
        "density going to zero and one rule, and that density is itself a continuation of the law.",
        "",
        *figure_links("P_sync", figures),
    ]
