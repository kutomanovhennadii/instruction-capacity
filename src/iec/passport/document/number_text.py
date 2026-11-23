"""The numbers of the text of the passport: a true minus sign, small parameters of laws by their significant digits and a
sensible precision of a capacity."""

from __future__ import annotations

import math

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
#: Below this size a small parameter is written by a power of ten rather than by a long row of zeros.
_POWER_BELOW = 1e-4


def number(value: float, digits: int = 2) -> str:
    """Write a number with a true minus sign.

    Args:
        value: The number.
        digits: The digits after the point.
    """
    return f"{value:.{digits}f}".replace("-", "−")


def parameter(value: float, digits: int = 2) -> str:
    """Write a parameter of a law or its half-width so that a small number never reads as zero.

    A number not below one unit of the last of ``digits`` places is written as ``number`` writes it; a smaller one that is
    not zero keeps two significant digits: ``0.00065`` stays ``0.00065`` and ``0.0019853`` becomes ``0.0020``, and below
    ``1e-4`` the number goes by a power of ten, ``1.2·10⁻⁶``.

    Args:
        value: The parameter.
        digits: The digits after the point for a number that is not small.
    """
    size = abs(value)
    if size == 0.0 or size >= 10.0**-digits or not math.isfinite(value):
        return number(value, digits)
    if size < _POWER_BELOW:
        mantissa, power = f"{value:.1e}".split("e")
        return f"{mantissa.replace('-', '−')}·10{str(int(power)).translate(_SUPERSCRIPT)}"
    return number(value, 1 - math.floor(math.log10(size)))


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
