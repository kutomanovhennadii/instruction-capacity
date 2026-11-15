"""The choice between the established form of a law and its alternatives.

A block names its established form first and its alternatives after it. The established form
stays unless an alternative describes the same bins much better: by more than ``SWITCH_DELTA`` of
the quasi-likelihood criterion ``QAIC = χ²/ĉ + 2k`` (Burnham and Anderson 2002: a model behind the
best by more than 10 has essentially no support). ``χ²`` is the weighted sum of the squared
residuals over the bins kept by the clipping, ``k`` the number of the free parameters, ``ĉ`` the
overdispersion: ``χ²`` per degree of freedom of the most flexible converged form, not below one.
The bins scatter around any law more than their own errors allow, and the plain ``χ² + 2k`` would
replace the established form at the smallest bend; ``ĉ`` brings the residuals to their real scatter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: The margin of the criterion by which an alternative replaces the established form.
SWITCH_DELTA = 10.0
#: The price of a free parameter in the criterion.
PARAMETER_PRICE = 2.0


@dataclass(frozen=True, slots=True)
class Contender:
    """A converged form over the kept bins: its position in the block, ``χ²`` and free parameters."""

    position: int
    chi2: float
    free: int


def overdispersion(contenders: Sequence[Contender], points: int) -> float:
    """Return ``ĉ``: ``χ²`` per degree of freedom of the most flexible contender, not below one.

    A contender without a degree of freedom does not give it; with none left - one.
    """
    flexible = [item for item in contenders if points > item.free]
    if not flexible:
        return 1.0
    widest = max(flexible, key=lambda item: (item.free, -item.position))
    return max(1.0, widest.chi2 / (points - widest.free))


def quasi_criterion(contender: Contender, c_hat: float) -> float:
    """Return ``QAIC = χ²/ĉ + 2k`` of a contender."""
    return contender.chi2 / c_hat + PARAMETER_PRICE * contender.free


def pick(
    contenders: Sequence[Contender], points: int, established: int | None
) -> tuple[int, float]:
    """Return the position of the chosen form and ``ĉ``.

    Args:
        contenders: The converged forms over the same bins, at least one.
        points: The number of the bins.
        established: The position of the established form when it is a contender; ``None`` - it
            is not, and the smallest criterion wins.
    """
    c_hat = overdispersion(contenders, points)
    best = min(contenders, key=lambda item: (quasi_criterion(item, c_hat), item.position))
    if established is None:
        return best.position, c_hat
    base = next(item for item in contenders if item.position == established)
    if quasi_criterion(base, c_hat) - quasi_criterion(best, c_hat) > SWITCH_DELTA:
        return best.position, c_hat
    return established, c_hat
