"""The axes of the folded design besides the grid: density, number of rules, rule number.

An axis is a strategy. It knows the bin of a trial, the coordinate whose weighted mean over the
trials of a bin is the abscissa of the bin, the bounds of a bin in the units of the axis, the
reference bin that has no column, and the prefix of the names of its columns. Density and the
number of rules are axes of a request: they are constant inside a point and come from its
conditions. The rule number is an axis of a cell: it varies inside a point, and only the rule
words of H carry it.
"""

from __future__ import annotations

import math
from typing import Protocol

from iec.passport.trials.layout.axis_bins import (
    axis_density_bin,
    axis_density_p,
    axis_density_reference,
)
from iec.passport.trials.layout.bin_layout import BinLayout

#: The reference bin of the number of rules and of the rule number: bin 1, a single rule.
RULE_REFERENCE_BIN = 1


class DesignAxis(Protocol):
    """An axis of the folded design: the names of its columns, its reference bin and its bins."""

    @property
    def letter(self) -> str:
        """The letter of the axis in the passport: ``P``, ``M`` or ``R``."""
        ...

    @property
    def prefix(self) -> str:
        """The prefix of the column names of the bins, as in the passport."""
        ...

    @property
    def coordinate(self) -> str:
        """The name of the coordinate summed over the trials of a bin."""
        ...

    def reference(self, layout: BinLayout) -> int:
        """Return the reference bin of the axis: the hidden zero, without a column."""
        ...

    def bounds(self, number: int, layout: BinLayout) -> tuple[float, float]:
        """Return the bounds of a bin in the units of the axis."""
        ...

    def abscissa(self, mean: float) -> dict[str, float]:
        """Return the abscissa of a bin from the weighted mean of its coordinate."""
        ...


class RequestAxis(DesignAxis, Protocol):
    """An axis constant inside a point: its bin and coordinate come from the conditions."""

    def bin_of(self, layout: BinLayout, D: int, m: int) -> int:  # noqa: N803 - protocol axes
        """Return the bin of a point with ``D`` distinct words and ``m`` rules."""
        ...

    def coordinate_sum(self, D: int, m: int, trials: int) -> float:  # noqa: N803 - protocol axes
        """Return the sum of the coordinate over ``trials`` trials of a point."""
        ...


class CellAxis(DesignAxis, Protocol):
    """An axis that varies inside a point: its bin and coordinate come from a value of a cell."""

    def bin_of(self, layout: BinLayout, value: int) -> int:
        """Return the bin of a value of a cell."""
        ...

    def coordinate_sum(self, value: int, trials: int) -> float:
        """Return the sum of the coordinate over ``trials`` trials of a cell."""
        ...


class DensityAxis:
    """Density ``p = m/D``: bins from the sparse end (1) to ``p = 1``, the reference bin of the
    start step, the mean density as the abscissa."""

    letter = "P"
    prefix = "p_bin_"
    coordinate = "p"

    def bin_of(self, layout: BinLayout, D: int, m: int) -> int:  # noqa: N803 - protocol axes
        """Return the density bin of a point."""
        return axis_density_bin(layout, D, m)

    def coordinate_sum(self, D: int, m: int, trials: int) -> float:  # noqa: N803 - protocol axes
        """Return the density summed over the trials of a point."""
        return m / D * trials

    def reference(self, layout: BinLayout) -> int:
        """Return the density bin of the start step."""
        return axis_density_reference(layout)

    def bounds(self, number: int, layout: BinLayout) -> tuple[float, float]:
        """Return the densities that bound a bin."""
        return axis_density_p(layout, number)

    def abscissa(self, mean: float) -> dict[str, float]:
        """Return the mean density of a bin."""
        return {"p": mean}


class RuleCountAxis:
    """The number of rules ``m``: doubling bins from one, the reference bin 1, the geometric mean
    as the abscissa."""

    letter = "M"
    prefix = "m_bin_"
    coordinate = "log_m"

    def bin_of(self, layout: BinLayout, D: int, m: int) -> int:  # noqa: N803 - protocol axes
        """Return the bin of the number of rules of a point."""
        return layout.row_bin(m)

    def coordinate_sum(self, D: int, m: int, trials: int) -> float:  # noqa: N803 - protocol axes
        """Return the logarithm of the number of rules summed over the trials of a point."""
        return math.log(m) * trials

    def reference(self, layout: BinLayout) -> int:
        """Return the reference bin: a single rule."""
        return RULE_REFERENCE_BIN

    def bounds(self, number: int, layout: BinLayout) -> tuple[float, float]:
        """Return the numbers of rules that bound a bin."""
        return _rule_bounds(number, layout)

    def abscissa(self, mean: float) -> dict[str, float]:
        """Return the geometric mean of the number of rules of a bin."""
        return {"m": math.exp(mean)}


class RuleNumberAxis:
    """The rule number: doubling bins from one, pooled past the last, the reference bin 1, the
    geometric mean as the abscissa."""

    letter = "R"
    prefix = "r_bin_"
    coordinate = "log_r"

    def bin_of(self, layout: BinLayout, value: int) -> int:
        """Return the bin of a rule number."""
        return layout.row_bin(value)

    def coordinate_sum(self, value: int, trials: int) -> float:
        """Return the logarithm of the rule number summed over the trials of a cell."""
        return math.log(value) * trials

    def reference(self, layout: BinLayout) -> int:
        """Return the reference bin: the first rule."""
        return RULE_REFERENCE_BIN

    def bounds(self, number: int, layout: BinLayout) -> tuple[float, float]:
        """Return the rule numbers that bound a bin."""
        return _rule_bounds(number, layout)

    def abscissa(self, mean: float) -> dict[str, float]:
        """Return the geometric mean of the rule number of a bin."""
        return {"r": math.exp(mean)}


def _rule_bounds(number: int, layout: BinLayout) -> tuple[float, float]:
    """Return the bounds of a bin of rules in rules; a number past the last bin gives the last."""
    lower, upper = layout.row_bounds[min(number, layout.R) - 1]
    return float(lower), float(upper)


#: The axes of the folded design.
DENSITY = DensityAxis()
RULE_COUNT = RuleCountAxis()
RULE_NUMBER = RuleNumberAxis()
_AXES: tuple[DesignAxis, ...] = (DENSITY, RULE_COUNT, RULE_NUMBER)
#: Prefixes of the axis column names - as in the passport.
COLUMN_PREFIX: dict[str, str] = {axis.letter: axis.prefix for axis in _AXES}
