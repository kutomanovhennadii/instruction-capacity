"""Folded design matrices of the quantities in the layout of the passport.

Text length and word number form one grid of "length × word number" cells: their contributions do
not add up (the article, section 5.2). Density, number of rules and rule number are bins with the
hidden zero in the reference bin. The columns are ``vs_V_S``, ``p_bin_k``, ``m_bin_k``, ``r_bin_k``,
the reference bins are for the density the bin of the starting step (``axis_density_reference``),
for the number of rules and the rule number bin 1. The grid cells carry the level, all of them
free. So the bins can carry smooth laws and the correction that gives the level.

A row of a folded matrix is a point and a grid cell, for H also a rule-number bin: the conditions
of the request are common to all trials of the row, and the folding is exact. The word-number
bin of a cell is the doubling of the line number, not past the last cell of the length bin. P_sync
takes the grid, P and M here, and its section drops M: an insertion has no rule number. H takes
VS-PMR (the grid, P, M and R).

What differs between the quantities - their trials of a point and their axes - is a strategy of
``quantity_designs`` chosen by the name of the quantity, and every axis is a strategy of
``design_axes``; the folding itself is one for all quantities.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.design.design_axes import DesignAxis
from iec.passport.trials.design.quantity_designs import QUANTITY_DESIGNS
from iec.passport.trials.layout.axis_bins import vs_grid_cells, vs_grid_tops
from iec.passport.trials.layout.bin_layout import REFERENCE_TOTAL_WORDS, BinLayout
from iec.passport.trials.place_trials import PointTrials

_WEIGHT_CODE = "POINT_WEIGHT_MISSING"
_VARIABLE_CODE = "VARIABLE_UNKNOWN"


@dataclass(frozen=True, slots=True)
class GridCell:
    """A cell of the "length × word number" grid with its trials.

    ``name`` - the passport column name ``vs_V_S``; ``lower_words`` and ``upper_words`` - the
    bounds of the length bin in words; ``points_with_failures`` - requests that gave failures in
    the cell.
    """

    v_bin: int
    s_bin: int
    name: str
    lower_words: int
    upper_words: int
    trials: int
    failures: int
    points_with_failures: int


@dataclass(frozen=True, slots=True)
class AxisBin:
    """A bin of axis P, M or R with trials: number, bounds in axis units, column name, counts."""

    number: int
    lower: float
    upper: float
    name: str
    trials: int
    failures: int
    points_with_failures: int


@dataclass(frozen=True, slots=True)
class AxisTable:
    """The bins of one axis with trials; ``reference`` - the passport reference bin, no column."""

    letter: str
    bins: tuple[AxisBin, ...]
    reference: int


@dataclass(frozen=True, slots=True, eq=False)
class VariableDesign:
    """The folded matrix of one quantity in the layout of the passport.

    ``x`` - indicators ``(rows, columns)``; ``failures`` and ``trials`` - failures and trials of
    a row; ``run`` - the number of the row's point in ``point_keys``, non-decreasing; ``names`` -
    passport column names; ``cells`` - grid cells with trials; ``axes`` - axes P, M and, for H,
    R; ``abscissae`` - centres of mass of cells and bins in the format of ``bin_abscissae`` of
    the passport, reference bins too.
    """

    variable: str
    x: np.ndarray
    failures: np.ndarray
    trials: np.ndarray
    run: np.ndarray
    point_keys: tuple[str, ...]
    names: tuple[str, ...]
    cells: tuple[GridCell, ...]
    axes: dict[str, AxisTable]
    abscissae: dict[str, dict[str, float]]


@dataclass
class _Tally:
    """Accumulator of a cell or a bin: counts, requests with failures, weighted coordinate sums."""

    trials: int = 0
    failures: int = 0
    weight: float = 0.0
    sums: dict[str, float] | None = None
    points: set[str] | None = None

    def add(self, key: str, trials: int, failures: int, weight: float, **sums: float) -> None:
        """Add trials; ``sums`` are sums of a coordinate over the trials, not means."""
        self.trials += trials
        self.failures += failures
        self.weight += weight * trials
        if self.sums is None:
            self.sums = {}
        for name, value in sums.items():
            self.sums[name] = self.sums.get(name, 0.0) + weight * value
        if failures:
            if self.points is None:
                self.points = set()
            self.points.add(key)

    def mean(self, name: str) -> float:
        """Return the mean of a coordinate over the trials with request weights."""
        if not self.sums or self.weight <= 0.0:
            return 0.0
        return self.sums.get(name, 0.0) / self.weight


def point_weight(weights: Mapping[str, float], key: str) -> float:
    """Return the weight of a request; a missing weight is an error, not a silent one.

    Raises:
        ContractViolationError: ``POINT_WEIGHT_MISSING``.
    """
    if key not in weights:
        raise ContractViolationError(f"point {key} has no weight", error_code=_WEIGHT_CODE)
    return float(weights[key])


def build_design(
    points: Sequence[PointTrials],
    variable: str,
    layout: BinLayout,
    weights: Mapping[str, float],
) -> VariableDesign:
    """Fold the trials of a quantity into a matrix in the layout of the passport.

    Args:
        points: The trials of the points.
        variable: ``P_sync``, ``H`` or ``F``.
        layout: The bin layout of the passport (``build_layout``).
        weights: Request weights by key (``cell_balanced_weights``).

    Returns:
        ``VariableDesign``. A point without trials of the quantity gives no rows (for F - the
        points with ``p = 1``).

    Raises:
        ContractViolationError: ``VARIABLE_UNKNOWN`` - an unknown quantity;
            ``POINT_WEIGHT_MISSING`` - a point with trials has no weight.
    """
    design = QUANTITY_DESIGNS.get(variable)
    if design is None:
        raise ContractViolationError(f"unknown variable {variable!r}", error_code=_VARIABLE_CODE)
    axes_of: tuple[DesignAxis, ...] = (*design.request_axes, *design.cell_axes)
    tops = vs_grid_tops(layout)
    rows: list[tuple[int, str, dict[str, int], int, int]] = []
    keys: list[str] = []
    cell_tally: dict[tuple[int, int], _Tally] = defaultdict(_Tally)
    axis_tally: dict[str, dict[int, _Tally]] = {
        axis.letter: defaultdict(_Tally) for axis in axes_of
    }
    for point in points:
        cells = design.cells(point)
        point_trials_count = sum(cell[2] for cell in cells)
        if point_trials_count == 0:
            continue
        weight = point_weight(weights, point.key)
        v = point.D * point.n
        v_bin = layout.volume_bin(v)
        u = math.log2(v / REFERENCE_TOTAL_WORDS)
        request = {
            axis.letter: axis.bin_of(layout, point.D, point.m) for axis in design.request_axes
        }
        run = len(keys)
        keys.append(point.key)
        merged: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(lambda: [0, 0])
        for s_bin, values, trials, failures, log_sum in cells:
            s_capped = min(s_bin, tops[v_bin - 1])
            cell_tally[(v_bin, s_capped)].add(
                point.key, trials, failures, weight, log_x=log_sum, u=u * trials
            )
            cell_bins: list[int] = []
            for cell_axis, value in zip(design.cell_axes, values, strict=True):
                number = cell_axis.bin_of(layout, value)
                axis_tally[cell_axis.letter][number].add(
                    point.key,
                    trials,
                    failures,
                    weight,
                    **{cell_axis.coordinate: cell_axis.coordinate_sum(value, trials)},
                )
                cell_bins.append(number)
            entry = merged[(s_capped, tuple(cell_bins))]
            entry[0] += trials
            entry[1] += failures
        for (s_capped, cell_key), (trials, failures) in sorted(merged.items()):
            row_bins = dict(request)
            row_bins.update(
                zip((axis.letter for axis in design.cell_axes), cell_key, strict=True)
            )
            rows.append((run, f"vs_{v_bin}_{s_capped}", row_bins, trials, failures))
        point_failures = sum(cell[3] for cell in cells)
        for request_axis in design.request_axes:
            coordinate = request_axis.coordinate_sum(point.D, point.m, point_trials_count)
            axis_tally[request_axis.letter][request[request_axis.letter]].add(
                point.key,
                point_trials_count,
                point_failures,
                weight,
                **{request_axis.coordinate: coordinate},
            )

    present = [cell for cell in vs_grid_cells(layout) if cell in cell_tally]
    cells_out = tuple(
        GridCell(
            v_bin=v_bin,
            s_bin=s_bin,
            name=f"vs_{v_bin}_{s_bin}",
            lower_words=layout.volume_bounds[v_bin - 1][0],
            upper_words=layout.volume_bounds[v_bin - 1][1],
            trials=cell_tally[(v_bin, s_bin)].trials,
            failures=cell_tally[(v_bin, s_bin)].failures,
            points_with_failures=len(cell_tally[(v_bin, s_bin)].points or ()),
        )
        for v_bin, s_bin in present
    )
    axes = {axis.letter: _axis_table(axis, axis_tally[axis.letter], layout) for axis in axes_of}
    axis_names = [
        item.name
        for axis in axes_of
        for item in axes[axis.letter].bins
        if item.number != axes[axis.letter].reference
    ]
    names = (*(cell.name for cell in cells_out), *axis_names)
    index = {name: position for position, name in enumerate(names)}
    prefixes = {axis.letter: axis.prefix for axis in axes_of}
    x = np.zeros((len(rows), len(names)))
    for row_number, (_, cell_name, row_bins, _, _) in enumerate(rows):
        x[row_number, index[cell_name]] = 1.0
        for letter, number in row_bins.items():
            column = index.get(f"{prefixes[letter]}{number}")
            if column is not None:
                x[row_number, column] = 1.0
    abscissae: dict[str, dict[str, float]] = {}
    for cell in cells_out:
        tally = cell_tally[(cell.v_bin, cell.s_bin)]
        abscissae[cell.name] = {"x": 2.0 ** tally.mean("log_x"), "u": tally.mean("u")}
    for axis in axes_of:
        for item in axes[axis.letter].bins:
            tally = axis_tally[axis.letter][item.number]
            abscissae[item.name] = axis.abscissa(tally.mean(axis.coordinate))
    return VariableDesign(
        variable=variable,
        x=x,
        failures=np.array([row[4] for row in rows], dtype=float),
        trials=np.array([row[3] for row in rows], dtype=float),
        run=np.array([row[0] for row in rows], dtype=np.int64),
        point_keys=tuple(keys),
        names=names,
        cells=cells_out,
        axes=axes,
        abscissae=abscissae,
    )


def _axis_table(axis: DesignAxis, tallies: Mapping[int, _Tally], layout: BinLayout) -> AxisTable:
    """Build the table of an axis with the passport reference bin.

    Args:
        axis: The axis.
        tallies: The accumulators of the bins of the axis by bin number.
        layout: The bin layout of the passport.

    Returns:
        ``AxisTable`` of the bins with trials, in the order of their numbers.
    """
    bins: list[AxisBin] = []
    for number in sorted(tallies):
        tally = tallies[number]
        if tally.trials == 0:
            continue
        lower, upper = axis.bounds(number, layout)
        bins.append(
            AxisBin(
                number=number,
                lower=lower,
                upper=upper,
                name=f"{axis.prefix}{number}",
                trials=tally.trials,
                failures=tally.failures,
                points_with_failures=len(tally.points or ()),
            )
        )
    return AxisTable(letter=axis.letter, bins=tuple(bins), reference=axis.reference(layout))
