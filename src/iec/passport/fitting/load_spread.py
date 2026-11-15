"""The spread of the request shift by the load: ``b_j ~ N(0, σ_j²)``, ``ln σ_j = z_j · γ``.

The row ``z_j`` of a request is a level and the free bins of its text length, density and number
of rules. A rare edge of an axis is merged with its neighbour until the group has
``ENOUGH_FAILING`` requests with failures; the most populated group of an axis is its reference
and has no column. The fit and the information of a node read the group of the bin of a request.
At a load that is not a bin ``ln σ`` goes by ``log₂`` of the axis between the centres of the
groups, stays at the edge group past them, and an axis that is not named is averaged over the
requests.

The fit by the load is made of steps that the chain of fits of a quantity also takes one by one:
the design of the spread (``spread_design``), its start (``spread_start``) and the law of a usable
fit of the spread (``load_fit``); ``fit_by_load`` makes them in a row for one fit.

A fit of the spread that stops on the floor - every halving of its step takes the spread of some
request below ``LOG_SIGMA_FLOOR`` - has a group whose variance goes to zero. Then the group of the
lowest parameter of the spread is merged with its neighbour on its axis and the fit is repeated
from the same start (``merge_floored_group``, ``fit_spread_merging``): the passport, the model of
rule execution and the model of the loss of place of the purchase merge by this one rule
(decision 4 of the passport standard). Fits that converge are not touched.

The share of the shift at a load, ``rho_of_load`` and ``load_rho``, is not used by the passport: it
serves a planner of points, whose information of a node depends on it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

from iec.passport.fitting.acceleration.folded_rows import FoldedDesign
from iec.passport.fitting.acceleration.sparse_design import build_sparse_design
from iec.passport.fitting.glmm_fit import GlmmFit, GlmmFitter, SpreadFit
from iec.passport.fitting.logit_fit import fit_failure
from iec.passport.trials.layout.vpm_cube import density_level

#: Requests with failures that a group at the edge of an axis must hold.
ENOUGH_FAILING = 30
#: The variance of the logistic noise: the share of the shift is ``σ²/(σ² + π²/3)``.
LOGISTIC_VARIANCE = math.pi**2 / 3.0
#: A load: length, density, number of rules; ``None`` - the axis is averaged over the requests.
Load = tuple[float | None, float | None, float | None]
_AXES = 3
#: The names of the axes in the order of a load, for the notes.
AXIS_NAMES = ("length", "density", "number of rules")


class _Layout(Protocol):
    """What the load of a request reads from a layout of the bins."""

    @property
    def P(self) -> int:  # noqa: N802 - the name of the model
        """Return the number of density bins."""
        ...

    def volume_bin(self, total_words: int) -> int:
        """Return the bin of a text length in words."""
        ...

    def row_bin(self, rule_index: int) -> int:
        """Return the bin of a rule number; the load reads it for the number of rules."""
        ...


class _Request(Protocol):
    """A request with its conditions."""

    @property
    def key(self) -> str:
        """Return the key of the request."""
        ...

    @property
    def D(self) -> int:  # noqa: N802 - protocol axis name
        """Return the number of distinct words of its text."""
        ...

    @property
    def n(self) -> int:
        """Return how many times every word occurs in its text."""
        ...

    @property
    def m(self) -> int:
        """Return its number of rules."""
        ...


@dataclass(frozen=True, slots=True)
class RequestLoad:
    """The load of a request: the bins of the axes and the values ``(v, p, m)``."""

    bins: tuple[int, int, int]
    values: tuple[float, float, float]


def request_load(layout: _Layout, D: int, n: int, m: int) -> RequestLoad:  # noqa: N803
    """Return the load of the conditions ``(D, n, m)`` in a layout of the bins."""
    words = D * n
    return RequestLoad(
        bins=(layout.volume_bin(words), max(1, density_level(D, m) + layout.P), layout.row_bin(m)),
        values=(float(words), m / D, float(m)),
    )


def request_loads(points: Iterable[_Request], layout: _Layout) -> dict[str, RequestLoad]:
    """Return the loads of requests by their keys."""
    return {point.key: request_load(layout, point.D, point.n, point.m) for point in points}


@dataclass(frozen=True, slots=True)
class SpreadAxis:
    """An axis of the spread: its groups of bins in the order of the axis.

    ``groups`` - the numbers of the bins; ``columns`` - the column of ``γ`` of a group, ``None``
    for the reference group; ``centers`` - the geometric mean of the load of its requests;
    ``shares`` - its share of the requests; ``failing`` - its requests with failures.
    """

    groups: tuple[tuple[int, ...], ...]
    columns: tuple[int | None, ...]
    centers: tuple[float, ...]
    shares: tuple[float, ...]
    failing: tuple[int, ...]

    def slot(self, number: int) -> int:
        """Return the group of a bin; a bin absent from the fit goes to the nearest lower group."""
        firsts = [group[0] for group in self.groups]
        return max(bisect.bisect_right(firsts, number) - 1, 0)


def _between(centers: Sequence[float], value: float) -> np.ndarray:
    """Return the weights of the groups at a value.

    Between the centres the weights go by ``log₂``; past an edge the edge group takes them all.
    """
    logs = np.log2(np.asarray(centers, dtype=float))
    x = math.log2(value)
    weights = np.zeros(len(logs))
    if x <= logs[0]:
        weights[0] = 1.0
    elif x >= logs[-1]:
        weights[-1] = 1.0
    else:
        left = int(np.searchsorted(logs, x, side="right")) - 1
        share = (x - logs[left]) / (logs[left + 1] - logs[left])
        weights[left], weights[left + 1] = 1.0 - share, share
    return weights


@dataclass(frozen=True, slots=True)
class LoadSpread:
    """The law of the spread: ``γ`` and the three axes - length, density, number of rules."""

    gamma: tuple[float, ...]
    axes: tuple[SpreadAxis, SpreadAxis, SpreadAxis]

    def row_of_bins(self, bins: tuple[int, int, int]) -> np.ndarray:
        """Return the row ``z`` of a request by the groups of its bins."""
        row = np.zeros(len(self.gamma))
        row[0] = 1.0
        for axis, number in zip(self.axes, bins, strict=True):
            column = axis.columns[axis.slot(number)]
            if column is not None:
                row[column] = 1.0
        return row

    def row_at(self, load: Load) -> np.ndarray:
        """Return the row ``z`` at a load that is not a bin (the docstring of the module)."""
        row = np.zeros(len(self.gamma))
        row[0] = 1.0
        for axis, value in zip(self.axes, load, strict=True):
            weights = axis.shares if value is None else _between(axis.centers, value)
            for column, weight in zip(axis.columns, weights, strict=True):
                if column is not None:
                    row[column] += float(weight)
        return row

    def sigma2_of_bins(self, bins: tuple[int, int, int]) -> float:
        """Return the variance of the shift of a request by the groups of its bins."""
        return math.exp(2.0 * float(self.row_of_bins(bins) @ np.asarray(self.gamma)))


def _merged_groups(numbers: Sequence[int], failing: Mapping[int, int]) -> list[list[int]]:
    """Merge the rare bins at the edges of an axis until a group has enough failing requests."""
    groups = [[number] for number in sorted(set(numbers))]

    def count(group: Sequence[int]) -> int:
        """Return the failing requests of a group of bins."""
        return sum(failing.get(number, 0) for number in group)

    while len(groups) > 2 and count(groups[0]) < ENOUGH_FAILING:
        groups[1] = groups[0] + groups[1]
        groups.pop(0)
    while len(groups) > 2 and count(groups[-1]) < ENOUGH_FAILING:
        groups[-2] = groups[-2] + groups[-1]
        groups.pop()
    return groups


def spread_axes(
    loads: Sequence[RequestLoad],
    failed: np.ndarray,
    groups_of: Mapping[int, Sequence[Sequence[int]]] | None = None,
) -> tuple[SpreadAxis, SpreadAxis, SpreadAxis]:
    """Return the three axes of the spread of the requests of a fit.

    Args:
        loads: The loads of the requests in the order of the fit.
        failed: For every request, whether it has a failure.
        groups_of: The groups of bins of an axis by its position, given instead of the merging
            of the rare edges (``merge_floored_group``); ``None`` - every axis merges its edges.
    """
    axes: list[SpreadAxis] = []
    next_column = 1
    for position in range(_AXES):
        numbers = [load.bins[position] for load in loads]
        failing: dict[int, int] = {}
        for number, flag in zip(numbers, failed, strict=True):
            failing[number] = failing.get(number, 0) + int(flag)
        if groups_of is not None and position in groups_of:
            groups = [list(group) for group in groups_of[position]]
        else:
            groups = _merged_groups(numbers, failing)
        slot_of = {number: slot for slot, group in enumerate(groups) for number in group}
        slots = np.array([slot_of[number] for number in numbers])
        counts = np.bincount(slots, minlength=len(groups))
        reference = int(counts.argmax())
        columns: list[int | None] = []
        for slot in range(len(groups)):
            if slot == reference:
                columns.append(None)
            else:
                columns.append(next_column)
                next_column += 1
        values = np.array([load.values[position] for load in loads], dtype=float)
        axes.append(
            SpreadAxis(
                groups=tuple(tuple(group) for group in groups),
                columns=tuple(columns),
                centers=tuple(
                    math.exp(float(np.log(values[slots == slot]).mean()))
                    for slot in range(len(groups))
                ),
                shares=tuple(float(count) / len(loads) for count in counts),
                failing=tuple(sum(failing.get(number, 0) for number in group) for group in groups),
            )
        )
    return axes[0], axes[1], axes[2]


def spread_matrix(
    axes: tuple[SpreadAxis, SpreadAxis, SpreadAxis], loads: Sequence[RequestLoad]
) -> np.ndarray:
    """Return the rows ``z`` of the requests: ``(requests, 1 + free groups)``."""
    size = 1 + sum(1 for axis in axes for column in axis.columns if column is not None)
    empty = LoadSpread(tuple(0.0 for _ in range(size)), axes)
    return np.array([empty.row_of_bins(load.bins) for load in loads])


@dataclass(frozen=True, slots=True, eq=False)
class SpreadDesign:
    """The design of the spread of the requests of a fit: the axes and the rows ``z``.

    ``axes`` - the three axes of the spread (``spread_axes``); ``matrix`` - the rows ``z`` of the
    requests ``(requests, 1 + free groups)`` (``spread_matrix``).
    """

    axes: tuple[SpreadAxis, SpreadAxis, SpreadAxis]
    matrix: np.ndarray

    @property
    def grouped(self) -> bool:
        """Return whether some axis has several groups, so the spread can depend on the load."""
        return self.matrix.shape[1] > 1


def failing_requests(rows: FoldedDesign) -> np.ndarray:
    """Return for every request of a fit whether one of its rows has a failure."""
    failed = np.add.reduceat(
        np.asarray(rows.trials, dtype=float) - np.asarray(rows.successes, dtype=float),
        np.asarray(rows.starts),
    )
    result: np.ndarray = failed > 0.0
    return result


def spread_design(rows: FoldedDesign, loads: Sequence[RequestLoad]) -> SpreadDesign:
    """Return the design of the spread of the requests of a fit.

    A request is failing when one of its rows has a failure (``failing_requests``).

    Args:
        rows: The folded design of the fit.
        loads: The loads of the requests in the order of the rows.
    """
    axes = spread_axes(loads, failing_requests(rows))
    return SpreadDesign(axes, spread_matrix(axes, loads))


@dataclass(frozen=True, slots=True)
class FloorMerge:
    """A group of the spread merged with its neighbour because the fit stopped on the floor.

    ``axis`` - the position of the axis in a load (``AXIS_NAMES``); ``bins`` - the bins of the
    group; ``into`` - the bins of the neighbour it joined.
    """

    axis: int
    bins: tuple[int, ...]
    into: tuple[int, ...]

    @property
    def text(self) -> str:
        """Return the merge for a note: the axis and the bins of both groups."""
        return (
            f"the group of the {AXIS_NAMES[self.axis]} bins {list(self.bins)} is merged with "
            f"the bins {list(self.into)}: the fit with the spread by the load stopped on the floor "
            "of the spread"
        )


def merge_floored_group(
    design: SpreadDesign, gamma: np.ndarray, loads: Sequence[RequestLoad], failed: np.ndarray
) -> tuple[SpreadDesign, FloorMerge] | None:
    """Merge the group of the lowest parameter of the spread with its neighbour on its axis.

    Among the groups of the axes with several groups the one with the lowest parameter is taken:
    ``γ`` of its column, zero for the reference group of an axis, which has no column (the level
    ``γ_0`` is shared by all and is no group). The reference group may be the one whose variance
    goes to zero: then its free neighbours climb, and the level falls to the floor. A group at the
    edge of its axis merges with its only neighbour, as the rare edges merge; a group in the middle
    - with the neighbour that has fewer requests with failures, the lower one on a tie. The axes are
    built anew with the merged groups, and the most populated group of an axis is its reference
    again.

    Args:
        design: The design of the spread of the fit that stopped on the floor.
        gamma: The parameters of the spread where that fit stopped.
        loads: The loads of the requests in the order of the fit.
        failed: For every request, whether it has a failure (``failing_requests``).

    Returns:
        The new design and the merge, or ``None``: every axis has one group.
    """
    candidates = [
        (0.0 if column is None else float(gamma[column]), position, slot)
        for position, axis in enumerate(design.axes)
        if len(axis.groups) > 1
        for slot, column in enumerate(axis.columns)
    ]
    if not candidates:
        return None
    _, position, slot = min(candidates)
    axis = design.axes[position]
    last = len(axis.groups) - 1
    if slot == 0:
        neighbour = 1
    elif slot == last:
        neighbour = last - 1
    else:
        neighbour = slot - 1 if axis.failing[slot - 1] <= axis.failing[slot + 1] else slot + 1
    low, high = sorted((slot, neighbour))
    groups = [list(group) for group in axis.groups]
    merged = [*groups[:low], groups[low] + groups[high], *groups[high + 1 :]]
    groups_of: dict[int, Sequence[Sequence[int]]] = {
        index: current.groups for index, current in enumerate(design.axes)
    }
    groups_of[position] = merged
    axes = spread_axes(loads, failed, groups_of)
    merge = FloorMerge(position, axis.groups[slot], axis.groups[neighbour])
    return SpreadDesign(axes, spread_matrix(axes, loads)), merge


class _FloorAware(Protocol):
    """A fit of the spread that tells whether it stopped on the floor."""

    @property
    def floored(self) -> bool:
        """Return whether the fit stopped because the floor refused every halving of a step."""
        ...

    @property
    def gamma(self) -> np.ndarray:
        """Return the parameters of the spread where the fit stopped."""
        ...


FitT = TypeVar("FitT", bound=_FloorAware)


def fit_spread_merging(
    design: SpreadDesign,
    fit_once: Callable[[SpreadDesign], FitT],
    loads: Sequence[RequestLoad],
    failed: np.ndarray,
) -> tuple[SpreadDesign, FitT | None, tuple[FloorMerge, ...]]:
    """Fit the spread, merging the group of the lowest spread while the fit stops on the floor.

    ``fit_once`` fits the spread of a design from the same start every time - the fit with one
    variance; the passport and the purchase give their own fits. A fit that did not stop on the
    floor is the result, whether usable or not: the caller judges it.

    Args:
        design: The design of the spread, with several groups on some axis.
        fit_once: One fit of the spread of a design.
        loads: The loads of the requests in the order of the fit.
        failed: For every request, whether it has a failure (``failing_requests``).

    Returns:
        The last design, its fit and the merges made; the fit is ``None`` when the merges left
        every axis with one group, and the spread cannot depend on the load.
    """
    merges: list[FloorMerge] = []
    while True:
        found = fit_once(design)
        if not found.floored:
            return design, found, tuple(merges)
        step = merge_floored_group(design, found.gamma, loads, failed)
        if step is None:
            return design, found, tuple(merges)
        design, merge = step
        merges.append(merge)
        if not design.grouped:
            return design, None, tuple(merges)


def spread_start(common: GlmmFit, design: SpreadDesign) -> np.ndarray:
    """Return the start of ``γ``: the level of the fit with one variance, zero for the groups."""
    gamma_start = np.zeros(design.matrix.shape[1])
    gamma_start[0] = 0.5 * math.log(common.sigma2)
    return gamma_start


@dataclass(frozen=True, slots=True)
class LoadFit:
    """The fit with the spread by the load.

    ``sigma2`` - the variance averaged over the requests with their weights; ``sigma2_row`` - its
    covariance with the coefficients and its variance at the end; ``merges`` - the groups merged
    because the fit stopped on the floor (``fit_spread_merging``).
    """

    fit: SpreadFit
    spread: LoadSpread
    sigma2: float
    sigma2_row: np.ndarray
    merges: tuple[FloorMerge, ...] = ()


def load_fit(
    found: SpreadFit,
    design: SpreadDesign,
    weight: np.ndarray,
    merges: tuple[FloorMerge, ...] = (),
) -> LoadFit:
    """Return the fit by the load from a usable fit of the spread (``fit_failure`` gave ``None``).

    The variance of the result is averaged over the requests with their weights, and its
    covariance row goes from ``γ`` by the delta method.

    Args:
        found: The converged fit of the spread with finite estimates.
        design: The design of the spread it was fitted on.
        weight: The weights of the requests.
        merges: The groups merged on the way to this design.
    """
    local = np.exp(2.0 * (design.matrix @ found.gamma))
    share = np.asarray(weight, dtype=float) / float(np.sum(weight))
    gain = (2.0 * share * local) @ design.matrix
    size = len(found.theta)
    sigma2_row = np.append(
        found.covariance[:size, size:] @ gain, gain @ found.covariance[size:, size:] @ gain
    )
    spread = LoadSpread(tuple(float(value) for value in found.gamma), design.axes)
    return LoadFit(found, spread, float(share @ local), sigma2_row, merges)


def fit_by_load(
    rows: FoldedDesign, weight: np.ndarray, common: GlmmFit, loads: Sequence[RequestLoad]
) -> LoadFit | None:
    """Fit the spread by the load from the estimates of the fit with one variance.

    A fit that stops on the floor merges the group of the lowest spread and is repeated
    (``fit_spread_merging``); the sparse design of the rows is built once for the repeats.

    Args:
        rows: The folded design of the fit.
        weight: The weights of the requests.
        common: The converged fit with one variance on the same rows.
        loads: The loads of the requests in the order of the rows.

    Returns:
        The fit, or ``None``: every axis has one group, or the fit did not converge or gave
        non-finite estimates (``fit_failure``).
    """
    design = spread_design(rows, loads)
    if not design.grouped:
        return None
    fitter = GlmmFitter(build_sparse_design(rows))

    def fit_once(current: SpreadDesign) -> SpreadFit:
        """Fit the spread of a design from the fit with one variance."""
        return fitter.fit_spread(
            weight, common.theta, spread_start(common, current), current.matrix, common.modes
        )

    design, found, merges = fit_spread_merging(design, fit_once, loads, failing_requests(rows))
    if found is None:
        return None
    if fit_failure(found.converged, found.steps, found.theta, found.gamma) is not None:
        return None
    return load_fit(found, design, weight, merges)


def rho_of_load(rho: float, spread: LoadSpread | None, load: RequestLoad | None) -> float:
    """Return the share of the shift at a load; without a spread or a load - the share ``rho``."""
    if spread is None or load is None:
        return rho
    sigma2 = spread.sigma2_of_bins(load.bins)
    return sigma2 / (sigma2 + LOGISTIC_VARIANCE)


def load_rho(
    rho: float, spread: LoadSpread | None, layout: _Layout, D: int, n: int, m: int  # noqa: N803
) -> float:
    """Return the share of the shift at the conditions ``(D, n, m)``; without a spread - ``rho``."""
    if spread is None:
        return rho
    return rho_of_load(rho, spread, request_load(layout, D, n, m))
