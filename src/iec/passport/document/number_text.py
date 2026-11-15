"""The numbers of the text of the passport: a true minus sign and a sensible precision of a capacity."""

from __future__ import annotations

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def number(value: float, digits: int = 2) -> str:
    """Write a number with a true minus sign.

    Args:
        value: The number.
        digits: The digits after the point.
    """
    return f"{value:.{digits}f}".replace("-", "−")


def size(value: float) -> str:
    """Write a capacity: large numbers by a power of ten, the others with a sensible precision.

    Args:
        value: The capacity.
    """
    if value >= 1e5:
        mantissa, power = f"{value:.1e}".split("e")
        return f"{mantissa}·10{str(int(power)).translate(_SUPERSCRIPT)}"
    if value >= 100.0:
        return f"{value:,.0f}"
    return number(value, 1)
