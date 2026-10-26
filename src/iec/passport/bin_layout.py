"""Bin layout of the VS-PMR passport and the doubling rule of its axes.

A bin is an interval of an axis with one shared coefficient. The axes are cut by doubling
the upper bound: ``{1}, {2}, {3-4}, {5-8}, ...`` from one, or ``{1-20}, {21-40}, {41-80}, ...``
from the 20 words of the start step. Early values are told apart one by one, late values are
pooled: there are fewer data there.
"""

from __future__ import annotations

from dataclasses import dataclass

from iec.infrastructure.exceptions import ContractViolationError

#: Start step of the measurement protocol - the first node ``(D, n, m)`` of a measurement.
#: The purchase cube cuts its axes from it, the length bins start at its text, and the zero
#: of the passport surface lies on it.
START_D = 4
START_N = 5
START_M = 1
#: Words of the start-step text ``v₀ = D·n``: the top of the first length bin and the zero
#: of the length coordinate ``u = log₂(v/v₀)``.
REFERENCE_TOTAL_WORDS = START_D * START_N


@dataclass(frozen=True, slots=True)
class BinLayout:
    """Bins of the passport axes.

    ``volume_bounds`` - text length ``v = D·n`` by doubling from ``v₀``; ``density_bounds`` -
    the ratio ``⌈D/m⌉`` by doubling from one (bin 1 is ``p = 1``, bin ``k ≥ 2`` is
    ``p ∈ [2^−(k−1), 2^−(k−2))``); ``row_bounds`` - the number of rules and the rule number by
    doubling from one; ``density_reference`` - the 1-based density bin (counted from
    ``p = 1``) that holds ``⌈D₀/m₀⌉`` of the start step: the implicit zero of the density
    block.
    """

    volume_bounds: tuple[tuple[int, int], ...]
    density_bounds: tuple[tuple[int, int], ...]
    row_bounds: tuple[tuple[int, int], ...]
    density_reference: int

    @property
    def R(self) -> int:  # noqa: N802 - model notation: number of rule bins
        """Return the number of rule bins."""
        return len(self.row_bounds)

    @property
    def P(self) -> int:  # noqa: N802 - model notation: number of density bins
        """Return the number of density bins."""
        return len(self.density_bounds)

    def row_bin(self, rule_index: int) -> int:
        """Return the rule bin (1..R) of a rule number; past the last bin - the last bin.

        A number past the last bin is pooled into it: losing such trials is worse than
        coarsening them.

        Raises:
            ContractViolationError: ``BIN_LAYOUT_INVALID`` - the rule number is below one.
        """
        if rule_index < 1:
            raise ContractViolationError(
                f"rule index must be >= 1, got {rule_index}",
                error_code="BIN_LAYOUT_INVALID",
            )
        return locate_bin(rule_index, self.row_bounds)

    def volume_bin(self, total_words: int) -> int:
        """Return the length bin (1..V) of ``v = D·n``; past the last bin - the last bin.

        Raises:
            ContractViolationError: ``BIN_LAYOUT_INVALID`` - the length is below one.
        """
        if total_words < 1:
            raise ContractViolationError(
                f"total_words must be >= 1, got {total_words}",
                error_code="BIN_LAYOUT_INVALID",
            )
        return locate_bin(total_words, self.volume_bounds)


def doubling_bounds(base: int, upper: int) -> tuple[tuple[int, int], ...]:
    """Return bin bounds that double the **upper bound** up to ``upper``.

    The first bin ends at ``base``, every next one at twice the previous bound:
    ``{1}, {2}, {3-4}, {5-8}, ...`` from one and ``{1-20}, {21-40}, ...`` from twenty. The bound
    doubles, not the width, so the first bins stay one value wide. The last bin is cut at
    ``upper``.
    """
    bounds: list[tuple[int, int]] = []
    lower = 1
    boundary = base
    while lower <= upper:
        bounds.append((lower, min(boundary, upper)))
        lower = boundary + 1
        boundary *= 2
    return tuple(bounds)


def bin_width(bounds: tuple[tuple[int, int], ...], index: int) -> int:
    """Return the number of values of bin ``index`` (1-based): ``(5, 8)`` holds 4.

    A jump of level between neighbouring bins divided by the width gives the increment per
    one value; the sum of increments over a bin equals the jump.

    Raises:
        ContractViolationError: ``BIN_LAYOUT_INVALID`` - the index is outside the bounds.
    """
    if index < 1 or index > len(bounds):
        raise ContractViolationError(
            f"bin index {index} is outside 1..{len(bounds)}",
            error_code="BIN_LAYOUT_INVALID",
        )
    lower, upper = bounds[index - 1]
    return upper - lower + 1


def locate_bin(position: int, bounds: tuple[tuple[int, int], ...]) -> int:
    """Return the 1-based bin that holds the position; past the last bin - the last bin."""
    for index, (lower, upper) in enumerate(bounds, start=1):
        if lower <= position <= upper:
            return index
    return len(bounds)
