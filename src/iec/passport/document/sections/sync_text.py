"""The text of the section of P_sync in the passport: the fit, the laws, the level, the capacities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iec.passport.document.text_cells import (
    capacity_rows,
    figure_links,
    fit_line,
    law_sentence,
    slice_sentence,
    spread_lines,
    with_half,
)


def sync_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of P_sync.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = ["## P_sync: the answer word stands against its reference word", "", fit_line(section), ""]
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
        *capacity_rows(section["capacities"]),
        "",
        "The instruction capacity is the number of rules at which the law of the number of rules "
        "brings the logit to zero, and next to it the walk along the bins of that axis; by the "
        "law it is none when the fall of that law over its bins does not differ from zero. The fatigue "
        "capacity is the word at which the full law of the grid at the density going to zero and one "
        "rule reaches zero, on a text of the same length: by the curve of the law, as for `H`. Full "
        "density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − δ_P` the change of "
        "the law of the density from the density going to zero to `p = 1`; a factor below one means "
        "that full density lowers the odds of an error. The interval is `e^(−P̂(1) ∓ h)`, `h` the "
        "half-width of `P̂(1)`.",
        "",
        *figure_links("P_sync", figures),
    ]
