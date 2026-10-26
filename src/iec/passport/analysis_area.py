"""Area of analysis of a measurement: the purchase cube D × m × n and its boundary.

The cube cuts the axes by doubling from the start step of the protocol (``D = 4``, ``m = 1``,
``n = 5``) up to the largest executed point - status ``ok`` or ``na``. A cell is represented
by its upper corner, the heaviest node of the cell. The state of the representative is read
from the points measured at it:

- the provider refused it - at least three ``api_error`` points and no ``ok`` or ``na`` - ``na``;
- fewer than three ``ok`` or ``na`` points - no state;
- no ``ok`` point - ``na``;
- every ``ok`` point lost the text (a failure of the third kind) - ``collapse``;
- otherwise - ``open``.

A cell is a boundary cell when its representative collapsed or got ``na``; it is unreachable
when the representative has no state (its task cannot be built, the model limits refuse it,
or it was not measured). The passport is fitted on the points of the other, ordinary cells.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from iec.contracts import PointStatus
from iec.infrastructure.exceptions import PassportError
from iec.passport.bin_layout import START_D, START_M, START_N, doubling_bounds, locate_bin
from iec.passport.measured_points import MeasuredPoint

#: Points of a node needed before its state is read - the replicate norm of the protocol.
RUNS_PER_NODE = 3

STATE_OPEN = "open"
STATE_COLLAPSE = "collapse"
STATE_NA = "na"

#: Mark of an unreachable cell in the printed cube: no count can stand there.
MARK_UNREACHABLE = "x"

Bounds = tuple[tuple[int, int], ...]


class CellState(Enum):
    """State of a cube cell - one of three."""

    ORDINARY = "ordinary"
    BOUNDARY = "boundary"
    UNREACHABLE = "unreachable"


class Cell(NamedTuple):
    """Address of a cell: the bin numbers of the three axes, each from one."""

    d_bin: int
    m_bin: int
    n_bin: int


@dataclass(frozen=True, slots=True)
class CubeBounds:
    """Bin bounds of the three cube axes: distinct words ``D``, rules ``m``, repeats ``n``."""

    div: Bounds
    row: Bounds
    rep: Bounds


@dataclass(frozen=True, slots=True)
class AreaCube:
    """The purchase cube of a measurement: ``ok`` counts and states of the cells.

    A cell absent from ``states`` is ordinary.
    """

    bounds: CubeBounds
    counts: dict[Cell, int] = field(default_factory=dict)
    states: dict[Cell, CellState] = field(default_factory=dict)

    def cell_of(self, D: int, n: int, m: int) -> Cell:  # noqa: N803 - protocol axes
        """Return the cell of a node; a coordinate past the last bin goes to the last bin."""
        return Cell(
            d_bin=locate_bin(D, self.bounds.div),
            m_bin=locate_bin(m, self.bounds.row),
            n_bin=locate_bin(n, self.bounds.rep),
        )

    def count_of(self, cell: Cell) -> int:
        """Return the number of ``ok`` points of a cell."""
        return self.counts.get(cell, 0)

    def state_of(self, cell: Cell) -> CellState:
        """Return the state of a cell."""
        return self.states.get(cell, CellState.ORDINARY)


@dataclass(frozen=True, slots=True)
class _NodeTally:
    """Points of one node by status and the ``ok`` points that lost the text."""

    ok: int = 0
    na: int = 0
    refused: int = 0
    kind_3: int = 0


def _tallies(points: Sequence[MeasuredPoint]) -> dict[tuple[int, int, int], _NodeTally]:
    """Count the points of every node ``(D, n, m)``."""
    counts: dict[tuple[int, int, int], list[int]] = {}
    for point in points:
        entry = counts.setdefault((point.D, point.n, point.m), [0, 0, 0, 0])
        if point.status is PointStatus.OK:
            entry[0] += 1
            if point.first_fail_kind_3_line is not None:
                entry[3] += 1
        elif point.status is PointStatus.NA:
            entry[1] += 1
        elif point.status is PointStatus.API_ERROR:
            entry[2] += 1
    return {
        node: _NodeTally(ok=ok, na=na, refused=refused, kind_3=kind_3)
        for node, (ok, na, refused, kind_3) in counts.items()
    }


def measured_state(tally: _NodeTally | None) -> str | None:
    """Return the state of a node from its points, or ``None`` when it has no state yet."""
    if tally is None:
        return None
    if tally.ok + tally.na == 0 and tally.refused >= RUNS_PER_NODE:
        return STATE_NA
    if tally.ok + tally.na < RUNS_PER_NODE:
        return None
    if tally.ok == 0:
        return STATE_NA
    if tally.kind_3 == tally.ok:
        return STATE_COLLAPSE
    return STATE_OPEN


def build_area_cube(points: Sequence[MeasuredPoint]) -> AreaCube:
    """Build the purchase cube of a measurement from its points.

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - no point was executed (``ok`` or ``na``).
    """
    executed = [p for p in points if p.status in (PointStatus.OK, PointStatus.NA)]
    if not executed:
        raise PassportError(
            "the measurement has no executed points (status ok or na): there is no area "
            "to analyse",
            error_code="PASSPORT_INPUT_MISSING",
        )
    bounds = CubeBounds(
        div=doubling_bounds(START_D, max(p.D for p in executed)),
        row=doubling_bounds(START_M, max(p.m for p in executed)),
        rep=doubling_bounds(START_N, max(p.n for p in executed)),
    )
    tallies = _tallies(points)
    states: dict[Cell, CellState] = {}
    for d_index, (_, d_top) in enumerate(bounds.div, start=1):
        for m_index, (_, m_top) in enumerate(bounds.row, start=1):
            for n_index, (_, n_top) in enumerate(bounds.rep, start=1):
                state = measured_state(tallies.get((d_top, n_top, m_top)))
                cell = Cell(d_bin=d_index, m_bin=m_index, n_bin=n_index)
                if state in (STATE_COLLAPSE, STATE_NA):
                    states[cell] = CellState.BOUNDARY
                elif state is None:
                    states[cell] = CellState.UNREACHABLE
    empty = AreaCube(bounds=bounds, states=states)
    counts = Counter(
        empty.cell_of(p.D, p.n, p.m) for p in points if p.status is PointStatus.OK
    )
    return AreaCube(bounds=bounds, counts=dict(counts), states=states)


def safe_points(points: Sequence[MeasuredPoint], cube: AreaCube) -> tuple[MeasuredPoint, ...]:
    """Return the points of the ordinary cells - the area where the passport is fitted."""
    return tuple(
        point
        for point in points
        if cube.state_of(cube.cell_of(point.D, point.n, point.m)) is CellState.ORDINARY
    )


def render_cube(cube: AreaCube) -> str:
    """Print the cube as slices by ``m``: a matrix per rule bin, rows ``D``, columns ``n``.

    A cell shows the number of ``ok`` points; an unreachable cell shows ``x``. Boundary
    cells are not marked: the boundary is read where the numbers meet the ``x``.
    """
    bounds = cube.bounds

    def label(pair: tuple[int, int]) -> str:
        lower, upper = pair
        return str(lower) if lower == upper else f"{lower}-{upper}"

    columns = [label(pair) for pair in bounds.rep]
    row_head = max((len(label(pair)) for pair in bounds.div), default=1)
    row_head = max(row_head, len("D\\n"))
    width = max((len(column) for column in columns), default=1)
    width = max(width, 4)

    lines: list[str] = ["bin cube: point counts by bins D × m × n"]
    for m_index, m_bin in enumerate(bounds.row, start=1):
        lines.append("")
        lines.append(f"m ∈ {label(m_bin)}")
        lines.append("D\\n".ljust(row_head) + "".join(column.rjust(width) for column in columns))
        for d_index, d_bin in enumerate(bounds.div, start=1):
            cells: list[str] = []
            for n_index in range(1, len(bounds.rep) + 1):
                cell = Cell(d_bin=d_index, m_bin=m_index, n_bin=n_index)
                if cube.state_of(cell) is CellState.UNREACHABLE:
                    text = MARK_UNREACHABLE
                else:
                    text = str(cube.count_of(cell))
                cells.append(text.rjust(width))
            lines.append(label(d_bin).ljust(row_head) + "".join(cells))
    return "\n".join(lines)
