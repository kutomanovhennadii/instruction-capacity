"""The capacity by the bins of the number of rules: the walk that spends the reserve of the level.

The bins of the number of rules are walked in the order of the axis. The price of a bin is its
coefficient minus the shift of the law, with the sign changed: what the bin costs against one rule.
The capacity is the number of rules at which the price reaches the reserve, with the share of the bin
at the crossing (the article, section 4.5); a reserve that the bins do not spend gives the edge of
the bins as a lower bound and what is left of the reserve. F and P_sync, which have no rule number,
walk so; H prices every rule along both axes (``rule_word_section.capacity_by_bins``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from iec.passport.laws.gls_tools import STATUS_MEASURED

#: A capacity whose reserve is not positive: the level does not hold at one rule.
STATUS_BELOW_ONE_RULE = "below_one_rule"
#: A capacity whose reserve the bins do not spend: only a lower bound is known.
STATUS_BEYOND_SUPPORT = "beyond_support"


def capacity_by_bin_walk(
    rows: Sequence[tuple[float, float]], delta_m: float, reserve: float
) -> dict[str, Any]:
    """Return the capacity by the bins of the number of rules.

    Args:
        rows: The centre and the raw coefficient of every bin of the law in the order of the axis;
            the reference bin has the coefficient zero.
        delta_m: The shift of the law of the number of rules, its value at one rule.
        reserve: The reserve of the level: the level minus the threshold of the logit.

    Returns:
        ``value`` and ``status``: ``measured``; ``below_one_rule`` without a value;
        ``beyond_support`` without a value, with ``lower_bound`` - the edge of the bins - and
        ``reserve_left``.
    """
    if reserve <= 0.0:
        return {"value": None, "status": STATUS_BELOW_ONE_RULE}
    edge, price = 1, 0.0
    for center, raw in rows:
        # The upper edge of the bin: the power of two not below its centre.
        upper = 2 ** math.ceil(math.log2(max(center, 1.0) - 1e-9)) if center > 1.0 else 1
        upper = max(upper, edge)
        current = -(raw - delta_m)
        if current >= reserve and current > price:
            value = edge + (reserve - price) / (current - price) * (upper - edge)
            return {"value": float(value), "status": STATUS_MEASURED}
        edge, price = upper, max(price, current)
    return {
        "value": None,
        "status": STATUS_BEYOND_SUPPORT,
        "lower_bound": edge,
        "reserve_left": reserve - price,
    }
