"""The choice among the forms of a law: the smallest criterion wins.

Every candidate form of a block is fitted through the same bins with the same weights, and the form
with the smallest quasi-likelihood criterion ``QAIC = χ²/ĉ + 2k`` describes them best (Burnham and
Anderson 2002) and is taken. ``χ²`` is the weighted sum of the squared residuals over the bins kept
by the clipping, ``k`` the number of the free parameters, ``ĉ`` the overdispersion: the smallest
``χ²`` per degree of freedom among the converged forms, not below one, so that a flexible form that
lies badly does not inflate it. The established form, the first of the block, wins a tie. An
alternative with a parameter held on a bound of its range does not replace the established form:
the data push it to the edge of its family, and its criterion says little of its shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: The price of a free parameter in the criterion.
PARAMETER_PRICE = 2.0
#: The position of the established form in its block.
ESTABLISHED = 0


@dataclass(frozen=True, slots=True)
class Contender:
    """A converged form over the kept bins: its position in the block, ``χ²``, its free parameters
    and whether a parameter is held on a bound of its range."""

    position: int
    chi2: float
    free: int
    held: bool = False


def overdispersion(contenders: Sequence[Contender], points: int) -> float:
    """Return ``ĉ``: the smallest ``χ²`` per degree of freedom among the contenders, not below one.

    A contender without a degree of freedom does not give it; with none left - one.
    """
    ratios = [item.chi2 / (points - item.free) for item in contenders if points > item.free]
    return max(1.0, min(ratios)) if ratios else 1.0


def quasi_criterion(contender: Contender, c_hat: float) -> float:
    """Return ``QAIC = χ²/ĉ + 2k`` of a contender."""
    return contender.chi2 / c_hat + PARAMETER_PRICE * contender.free


def pick(contenders: Sequence[Contender], points: int) -> tuple[int, float]:
    """Return the position of the chosen form and ``ĉ``.

    The smallest criterion wins and the earlier form a tie. With the established form among the
    contenders an alternative with a held parameter does not compete.

    Args:
        contenders: The converged forms over the same bins, at least one.
        points: The number of the bins.
    """
    c_hat = overdispersion(contenders, points)
    with_established = any(item.position == ESTABLISHED for item in contenders)
    eligible = [
        item
        for item in contenders
        if item.position == ESTABLISHED or not item.held or not with_established
    ]
    best = min(eligible, key=lambda item: (quasi_criterion(item, c_hat), item.position))
    return best.position, c_hat
