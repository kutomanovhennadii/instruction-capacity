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
from iec.passport.laws.law_rows import LEVEL, capacity_row, half_width

#: A capacity whose law does not fall: no root.
STATUS_NO_SLOPE = "no_slope"


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
