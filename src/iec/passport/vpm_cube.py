"""Run cube ``V × P × M`` of the passport: the cell of a point, the passport bins, weights.

Text length ``v = D·n``, density ``p = m/D`` and the number of rules ``m`` are constant inside
a point - they are chosen when the node is named. Word position and rule number vary inside
a point, and their distribution is set by the protocol. The cube of the first three axes is
the **run cube**: the passport takes its length, density and rule-count bins from it, and
the point weights balance its cells.

**The density axis is the density itself, from small values to one.** The ``P`` coordinate
of a cell is the **density level** ``⌊log₂ p⌋``: ``0`` - ``p = 1``, ``−1`` - ``[1/2, 1)``,
``−2`` - ``[1/4, 1/2)``, ... It is computed exactly in integers: the level is
``−⌈log₂ ⌈D/m⌉⌉``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.contracts import PointStatus
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport.bin_layout import (
    REFERENCE_TOTAL_WORDS,
    START_D,
    START_M,
    BinLayout,
    doubling_bounds,
    locate_bin,
)
from iec.passport.measured_points import MeasuredPoint

_INVALID_CODE = "RUN_CUBE_INVALID"

#: Upper bounds of doubling bins from one: 1, 2, 4, ..., 2^39 - with room for any axis.
_DOUBLINGS = 2 ** np.arange(40, dtype=np.int64)


def density_level(D: int, m: int) -> int:  # noqa: N803 - protocol axes
    """Return the density level ``⌊log₂(m/D)⌋`` of a node: ``0`` at ``p = 1``, then negative.

    Computed in integers: ``−⌈log₂ ⌈D/m⌉⌉ = −((⌈D/m⌉ − 1).bit_length())``.

    Raises:
        ContractViolationError: ``RUN_CUBE_INVALID`` - ``1 ≤ m ≤ D`` does not hold.
    """
    if m < 1 or m > D:
        raise ContractViolationError(
            f"density needs 1 <= m <= D, got D={D}, m={m}", error_code=_INVALID_CODE
        )
    return -((-(-D // m) - 1).bit_length())


def density_p(level: int) -> tuple[float, float]:
    """Return the density of a level: ``(1, 1)`` at ``0``, ``[2^ℓ, 2^(ℓ+1))`` at ``ℓ < 0``."""
    if level == 0:
        return (1.0, 1.0)
    return (2.0**level, 2.0 ** (level + 1))


@dataclass(frozen=True, slots=True)
class VpmCube:
    """Run cube ``V × P × M``: cell ``(length bin, density level, rule-count bin)``.

    ``reference_total_words`` is ``v₀``, the top of the first length bin. Length and
    rule-count bins count from one, the density level is ``0`` at ``p = 1`` and negative
    below.
    """

    reference_total_words: int

    def __post_init__(self) -> None:
        """Require a positive reference length."""
        if self.reference_total_words < 1:
            raise ContractViolationError(
                f"reference_total_words must be >= 1, got {self.reference_total_words}",
                error_code=_INVALID_CODE,
            )

    def cell_of(self, D: int, n: int, m: int) -> tuple[int, ...]:  # noqa: N803 - protocol axes
        """Return the cell ``(V, P, M)`` of a node.

        Raises:
            ContractViolationError: ``RUN_CUBE_INVALID`` - ``1 ≤ m ≤ D`` or ``n ≥ 1`` does not
                hold: the protocol does not allow such a node.
        """
        if n < 1 or m < 1 or m > D:
            raise ContractViolationError(
                f"a run needs n >= 1 and 1 <= m <= D, got D={D}, n={n}, m={m}",
                error_code=_INVALID_CODE,
            )
        row = self.cells_of(np.array([[D, n, m]], dtype=np.int64))[0]
        return (int(row[0]), int(row[1]), int(row[2]))

    def cells_of(self, nodes: np.ndarray) -> np.ndarray:
        """Return the cells of an ``(N, 3)`` array of nodes with columns ``D, n, m``.

        The length and rule-count bin is the first doubling upper bound not below the value;
        the density level is minus the number of the first power of two not below ``⌈D/m⌉``.
        """
        array = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
        words = array[:, 0] * array[:, 1]
        ratio = -(-array[:, 0] // array[:, 2])
        volume = np.searchsorted(self.reference_total_words * _DOUBLINGS, words, side="left")
        level = -np.searchsorted(_DOUBLINGS, ratio, side="left")
        rules = np.searchsorted(_DOUBLINGS, array[:, 2], side="left")
        return np.column_stack([volume + 1, level, rules + 1]).astype(np.int64)

    def layout_bounds(self, points: Sequence[MeasuredPoint]) -> dict[str, Any]:
        """Return the passport axis bounds over the ``ok`` points - from the cube, uncapped.

        Length up to the top of the bin of the longest text; density down to the sparsest
        level (as integer ratios ``⌈D/m⌉`` from ``p = 1``); rule count up to the top of the
        bin of the largest ``m`` (a rule number never exceeds ``m``). No ``ok`` point - an
        empty dictionary.
        """
        ok = [point for point in points if point.status is PointStatus.OK]
        if not ok:
            return {}
        cells = self.cells_of(np.array([[p.D, p.n, p.m] for p in ok], dtype=np.int64))
        volume_top = self.reference_total_words * 2 ** (int(cells[:, 0].max()) - 1)
        return {
            "volume_bounds": doubling_bounds(self.reference_total_words, volume_top),
            "density_bounds": doubling_bounds(1, 2 ** (-int(cells[:, 1].min()))),
            "row_bounds": doubling_bounds(1, 2 ** (int(cells[:, 2].max()) - 1)),
        }


def cell_balanced_weights(
    ok_points: Sequence[MeasuredPoint], cube: VpmCube
) -> tuple[dict[str, float], dict[str, Any]] | None:
    """Return point weights that make every cube cell weigh the same, and their summary.

    A model is an approximation, and a plain maximum likelihood hides its miss where it is
    cheapest - in the overfilled cells - at the cost of the free bins of other axes. A weight
    inverse to the occupancy of the point's cell moves the fit from the purchase distribution
    to the target one, an equal cell of the cube (importance weighting). Weights are
    normalised to the mean weight 1: a common factor changes neither the estimate nor the
    sandwich.

    Returns:
        The pair "weight by point key, summary", or ``None`` without points. Summary:
        ``cells`` - cells with points, ``runs`` - points, ``max_occupancy``, ``n_eff`` - the
        effective number of points ``(Σw)²/Σw²``, ``min_weight``, ``max_weight``.
    """
    if not ok_points:
        return None
    cells = {point.key: cube.cell_of(point.D, point.n, point.m) for point in ok_points}
    occupancy = Counter(cells.values())
    raw = {key: 1.0 / occupancy[cell] for key, cell in cells.items()}
    mean = sum(raw.values()) / len(raw)
    weights = {key: value / mean for key, value in raw.items()}
    total = sum(weights.values())
    squares = sum(value * value for value in weights.values())
    summary = {
        "cells": len(occupancy),
        "runs": len(weights),
        "max_occupancy": max(occupancy.values()),
        "n_eff": total * total / squares,
        "min_weight": min(weights.values()),
        "max_weight": max(weights.values()),
    }
    return weights, summary


def build_layout(points: Sequence[MeasuredPoint]) -> BinLayout:
    """Return the passport layout over the ``ok`` points: the bins of the run cube.

    The reference density bin - the implicit zero of the ``P`` block - is the bin of the
    start step, ``⌈D₀/m₀⌉ = 4`` (``p ∈ [1/4, 1/2)``).

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - there are no ``ok`` points.
    """
    bounds = VpmCube(REFERENCE_TOTAL_WORDS).layout_bounds(points)
    if not bounds:
        raise PassportError(
            "the area of analysis has no ok points: there are no bins to cut",
            error_code="PASSPORT_INPUT_MISSING",
        )
    density_bounds = bounds["density_bounds"]
    return BinLayout(
        volume_bounds=bounds["volume_bounds"],
        density_bounds=density_bounds,
        row_bounds=bounds["row_bounds"],
        density_reference=locate_bin(-(-START_D // START_M), density_bounds),
    )
