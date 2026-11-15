"""The text of the section of H in the passport: the fit, the laws, the level, the capacities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from iec.passport.document.number_text import number
from iec.passport.document.text_cells import (
    capacity_rows,
    figure_links,
    fit_line,
    law_sentence,
    on_bound,
    spread_lines,
    with_half,
)


def rule_word_lines(section: Mapping[str, Any], figures: Sequence[str]) -> list[str]:
    """Return the section of H.

    Args:
        section: The section of the quantity.
        figures: The names of the figures of the quantity by the plan of the figures.
    """
    lines = ["## H: a rule word on its place is replaced correctly", "", fit_line(section), ""]
    if section["status"] != "measured":
        return [*lines, "There are no laws: the fit did not converge.", ""]
    laws = section["laws"]
    grid = laws["grid"]
    return [
        *lines,
        *spread_lines(section),
        "The law of the grid: a length slice is the exponent `μ_v − A·2^(γ·log₂ s)`, which dies out to "
        "its level `μ_v` at `s → 0`, and the law of the levels runs through the levels of the slices. "
        f"`A = {number(grid['A'], 3)} ± {number(grid['half']['A'], 3)}`, "
        f"`γ = {number(grid['gamma'], 3)} ± {number(grid['half']['gamma'], 3)}`; "
        f"by the own variances, `χ² = {number(grid['chi2'], 1)}` over {grid['n_points']} cells"
        f"{on_bound(grid)}.",
        "",
        law_sentence("The law of the levels of the slices", laws["grid_plateau"]),
        "",
        law_sentence("The law of the density", laws["density"]),
        "",
        law_sentence("The law of the number of rules", laws["table_size"]),
        "",
        law_sentence("The law of the rule number", laws["rule_position"]),
        "",
        f"The level `α = μ₀ + μ_P + λ_M + λ_R = {with_half(section['alpha'])}`. The loss of the density "
        f"at `p = 1` is {number(section['density_loss_at_one'])} logits.",
        "",
        *capacity_rows(section["capacities"]),
        "",
        "Full density multiplies the odds of an error by `e^(−P̂(1))`, `P̂(1) = P(1) − μ_P` the change "
        "of the law of the density from its zero point, `p = 2⁻¹⁰` for the two-slope hyperbola, to "
        "`p = 1`; a factor below one means that full density lowers the odds of an error. The "
        "interval is `e^(−P̂(1) ∓ h)`, `h` the half-width of `P̂(1)`.",
        "",
        *figure_links("H", figures),
    ]
