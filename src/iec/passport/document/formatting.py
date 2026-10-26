"""Number formatting of the passport document: at least two decimals, ± 1.96·SE."""

from __future__ import annotations

import math
from typing import Any

from iec.passport.model_fitting import HALFWIDTH_QUANTILE

#: Blocks of the approximation: the key of the law, the key of its restored bins, the title.
APPROXIMATION_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("vs_grid", "vs_grid", "length × position grid"),
    ("density", "density_profile", "replacement density P"),
    ("table_size", "table_size_profile", "number of rules M"),
    ("rule_position", "rule_position_profile", "rule number R"),
)


def format_number(value: float, *, signed: bool = False) -> str:
    """Write a number with at least two decimals; below one - three significant digits
    (``67.01``, ``18.00``, ``0.392``, ``0.0123``). ``signed`` prints the sign of positives too."""
    sign = "+" if signed else ""
    if value == 0.0 or not math.isfinite(value):
        return f"{value:{sign}.2f}"
    decimals = max(2, 2 - math.floor(math.log10(abs(value))))
    return f"{value:{sign}.{decimals}f}"


def approx_value(value: Any, se: Any) -> str:
    """Write ``value ± 1.96·SE``; without SE - the value; without a number - a dash."""
    if value is None:
        return "—"
    if se is None:
        return format_number(float(value))
    return f"{format_number(float(value))} ± {format_number(HALFWIDTH_QUANTILE * float(se))}"


def approx_number(value: Any) -> str:
    """Write an optional number without an interval, a count as an integer, or a dash."""
    if value is None:
        return "—"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return format_number(float(value))


def approx_signed(value: Any) -> str:
    """Write a deviation with its sign, or a dash."""
    return "—" if value is None else format_number(float(value), signed=True)


def optional_number(value: Any) -> str:
    """Write an optional estimate; ``None`` is "not estimated", not zero."""
    return "not estimated" if value is None else format_number(float(value))


def weighting_line(weighting: Any) -> str:
    """Say whether the fit is weighted by the cells of the run cube and what it cost."""
    if not isinstance(weighting, dict):
        return "- weighting by the cells of the run cube: none"
    return (
        f"- weighting by the cells of the run cube V × P × M: {weighting.get('cells')} cells, "
        f"{weighting.get('runs')} points, largest occupancy {weighting.get('max_occupancy')}; "
        f"effective points n_eff = {optional_number(weighting.get('n_eff'))}"
    )
