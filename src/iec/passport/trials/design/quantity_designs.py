"""The designs of the three quantities: whose trials of a point and which axes besides the grid.

The design of a quantity is a strategy chosen by the name of the quantity from a registry. P_sync
takes the places of a point, H its rule words and F its background words. All three carry the
axes of a request - density and the number of rules; H carries also the rule number, an axis of a
cell. The folding of the matrix (``folded_design``) knows only this interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from iec.passport.trials.design.design_axes import (
    DENSITY,
    RULE_COUNT,
    RULE_NUMBER,
    CellAxis,
    RequestAxis,
)
from iec.passport.trials.place_trials import PointTrials

#: A cell of the trials of a point: (word-number bin, the values of the cell axes, trials,
#: failures, Σ log₂ s); the values follow the cell axes of the quantity.
DesignCell = tuple[int, tuple[int, ...], int, int, float]


class QuantityDesign(Protocol):
    """The design of one quantity: its trials of a point and its axes besides the grid."""

    @property
    def variable(self) -> str:
        """The name of the quantity: ``P_sync``, ``H`` or ``F``."""
        ...

    @property
    def request_axes(self) -> tuple[RequestAxis, ...]:
        """The axes constant inside a point."""
        ...

    @property
    def cell_axes(self) -> tuple[CellAxis, ...]:
        """The axes that vary inside a point."""
        ...

    def cells(self, point: PointTrials) -> list[DesignCell]:
        """Return the cells of the trials of the quantity in a point."""
        ...


class SyncDesign:
    """P_sync: every place of a point is a trial; the axes density and number of rules."""

    variable = "P_sync"
    request_axes: tuple[RequestAxis, ...] = (DENSITY, RULE_COUNT)
    cell_axes: tuple[CellAxis, ...] = ()

    def cells(self, point: PointTrials) -> list[DesignCell]:
        """Return the cells of the places of a point."""
        return _plain_cells(point.sync)


class RuleDesign:
    """H: a rule word on its place is a trial; the axes density, number of rules, rule number."""

    variable = "H"
    request_axes: tuple[RequestAxis, ...] = (DENSITY, RULE_COUNT)
    cell_axes: tuple[CellAxis, ...] = (RULE_NUMBER,)

    def cells(self, point: PointTrials) -> list[DesignCell]:
        """Return the cells of the rule words of a point with their rule numbers."""
        return [
            (s_bin, (rule_index,), trials, failures, log_sum)
            for s_bin, rule_index, trials, failures, log_sum in point.rule
        ]


class BackgroundDesign:
    """F: a background word on its place is a trial; the axes density and number of rules."""

    variable = "F"
    request_axes: tuple[RequestAxis, ...] = (DENSITY, RULE_COUNT)
    cell_axes: tuple[CellAxis, ...] = ()

    def cells(self, point: PointTrials) -> list[DesignCell]:
        """Return the cells of the background words of a point."""
        return _plain_cells(point.background)


def _plain_cells(rows: tuple[tuple[int, int, int, float], ...]) -> list[DesignCell]:
    """Return the cells of trials that carry no value of a cell axis.

    Args:
        rows: The trials of a point by word-number bins: (bin, trials, failures, Σ log₂ s).

    Returns:
        The cells with an empty tuple of cell-axis values.
    """
    return [(s_bin, (), trials, failures, log_sum) for s_bin, trials, failures, log_sum in rows]


_DESIGNS: tuple[QuantityDesign, ...] = (SyncDesign(), RuleDesign(), BackgroundDesign())
#: The designs of the quantities by name.
QUANTITY_DESIGNS: Mapping[str, QuantityDesign] = {design.variable: design for design in _DESIGNS}
#: The axes of every quantity besides the grid: model VSPM for P_sync and F, VS-PMR for H.
VARIABLE_AXES: dict[str, tuple[str, ...]] = {
    name: tuple(axis.letter for axis in (*design.request_axes, *design.cell_axes))
    for name, design in QUANTITY_DESIGNS.items()
}
