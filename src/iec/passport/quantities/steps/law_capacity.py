"""The capacity by the root of the laws: the load at which they bring the logit to its threshold.

A capacity is the load at which the laws bring the logit of the median request at the level
:data:`LEVEL` to its threshold. The laws are of any form (``laws.block_choice``): the section gives
the root as a function of the parameters of its laws, and the delta method carries its derivative
by the parameters to the coefficients of the fit through their rows (``LawStack``). Laws that do not
reach the threshold give no root: the row has no value and the status ``no_slope``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from iec.passport.laws.block_choice import LawStack
from iec.passport.laws.gls_tools import STATUS_MEASURED
from iec.passport.laws.law_rows import LEVEL, capacity_row, half_width

#: A capacity whose law does not fall: no root.
STATUS_NO_SLOPE = "no_slope"
#: The measure of the density: how many times full density multiplies the odds of an error.
DENSITY_ODDS_FACTOR = "density_odds_factor"
#: A measure past the measured zone of its law: a continuation of the law.
STATUS_CONDITIONAL = "conditional"


def no_slope_row(key: str) -> dict[str, Any]:
    """Return the row of a capacity by a law that gives no root.

    Args:
        key: The name of the capacity.
    """
    return {"key": key, "level": LEVEL, "method": "law", "value": None, "status": STATUS_NO_SLOPE}


def root_capacity_row(
    key: str,
    function: Callable[[np.ndarray], float],
    stack: LawStack,
    joint: np.ndarray,
    support: float,
    *,
    scale: float = 1.0,
    ceiling: float = math.inf,
) -> dict[str, Any]:
    """Return the capacity by the root of the laws of any form.

    ``function`` gives the logarithm of the capacity from a vector of the parameters of the laws,
    ``nan`` - the laws have no root; the half-width goes by the delta method over the stack.

    Args:
        key: The name of the capacity.
        function: The logarithm of the capacity from a vector of the stack.
        stack: The parameters of the laws and their rows over the coefficients.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        support: The edge of the measured region in the units of the capacity.
        scale: The unit of the logarithm: 20 words for a length counted from 20 words.
        ceiling: The largest logarithm the interval of the row may reach.

    Returns:
        The row of ``law_rows.capacity_row``, or :func:`no_slope_row` for laws without a root or a
        capacity whose interval passes ``ceiling``.
    """
    x = function(stack.vector)
    if math.isnan(x):
        return no_slope_row(key)
    half = half_width(stack.gradient(function), joint)
    if not math.isfinite(half) or x + half > ceiling:
        return no_slope_row(key)
    return capacity_row(key, LEVEL, x, half, support, scale=scale)


def odds_factor_row(
    change: Callable[[np.ndarray], float],
    stack: LawStack,
    joint: np.ndarray,
    *,
    conditional: bool = False,
) -> dict[str, Any]:
    """Return how many times full density multiplies the odds of an error: ``e^(−P̂(1))``.

    ``P̂(1)`` is the change of the logit of a success by the law of the density from its zero point
    to ``p = 1``; the odds of an error are the ratio of its probability to the probability of a
    success, and while errors are rare it is almost the factor of the probability of an error
    itself. The factor keeps the sign of the change: below one, full density lowers the odds of an
    error, as the rising law of the density of H does. The error goes as a factor:
    ``e^(−P̂ ∓ h)``, ``h`` the half-width of ``P̂(1)`` by the delta method (decision 7 of the passport
    standard: the measure stands instead of the guaranteed fatigue capacity; written as the factor
    with its interval since 12.11).

    Args:
        change: ``P̂(1)`` from a vector of the parameters of the laws.
        stack: The parameters of the laws and their rows over the coefficients.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        conditional: The measure continues the law of the density past its measured zone: at
            ``p = 1`` a text of F has no background words.

    Returns:
        The row ``density_odds_factor`` with ``value``, ``interval``, ``status`` and the change of
        the logit itself, ``change`` and ``change_half``.
    """
    value = change(stack.vector)
    half = half_width(stack.gradient(change), joint)
    return {
        "key": DENSITY_ODDS_FACTOR,
        "level": None,
        "method": "law",
        "value": math.exp(-value),
        "interval": [math.exp(-value - half), math.exp(-value + half)],
        "status": STATUS_CONDITIONAL if conditional else STATUS_MEASURED,
        "change": value,
        "change_half": half,
    }
