"""Forms of a loss that grows with the load: exponential, binary exponential, saturation, power.

The exponential law ``c + A·e^(−x/τ)`` falls from ``c + A`` at ``x = 0`` to ``c`` with the length
``τ``: the share of the answer of F. The binary exponential ``μ − A·2^(γx)`` goes over the
logarithm of a load, ``x = log₂ s``: the loss ``A·s^γ`` grows by the factor ``2^γ`` per doubling of
the load and vanishes with it, so the law tends to ``μ`` at ``x → −∞``; a length slice of the grid
of H. The saturation ``μ − A·(1 − e^(−x/x_c))`` starts at ``μ`` at ``x = 0``, and its loss
saturates at ``A``, reaching ``1 − e⁻¹`` of it at ``x = x_c``: the density law of the first
article. The power law ``λ − A·(x^γ − 1)`` is ``λ`` at ``x = 1`` and its loss grows as ``x^γ``: the
rule-number law of the first article. All four describe a loss, so ``A ≥ 0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The range of the exponent ``γ`` of a binary exponential, as for a length slice of H.
GAMMA_RANGE = (0.005, 2.0)
#: The range of the exponent ``γ`` of a power law.
POWER_RANGE = (1e-3, 10.0)
#: The range of a length of a law as shares of the span of the abscissae.
LENGTH_SHARES = (1e-3, 1e3)
#: The smallest span of the abscissae for the range of a length.
SPAN_FLOOR = 1e-6


def _span(x: np.ndarray) -> float:
    """Return the span of the abscissae, at least ``SPAN_FLOOR``."""
    return max(float(x.max() - x.min()), SPAN_FLOOR)


def _length_range(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the range of a law with a length: ``A ≥ 0``, the length in ``LENGTH_SHARES`` spans."""
    span = _span(x)
    lower = np.array([-np.inf, 0.0, LENGTH_SHARES[0] * span])
    upper = np.array([np.inf, np.inf, LENGTH_SHARES[1] * span])
    return lower, upper


@dataclass(frozen=True, slots=True)
class ExponentialForm:
    """The exponential law ``c + A·e^(−x/τ)``; ``A`` and ``τ`` may stop on a bound."""

    name = "exponential"
    parameters = ("c", "A", "tau")
    held_candidates = (1, 2)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``c + A·e^(−x/τ)``."""
        return params[0] + params[1] * np.exp(-x / params[2])

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(c, A, τ) = (1, e^(−x/τ), A·x/τ²·e^(−x/τ))``."""
        decay = np.exp(-x / params[2])
        return np.column_stack(
            [np.ones_like(x), decay, params[1] * x / (params[2] * params[2]) * decay]
        )

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the law is over a load in its own units, its shift is at ``x = 0``."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``c + A·e^(−x/τ)`` with the symbol of the abscissa."""
        return f"c + A·e^(−{symbol}/τ)"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A ≥ 0`` and ``τ`` from a thousandth to 1000 spans of the abscissae."""
        return _length_range(x)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``c`` and ``A`` from the ends of the axis, ``τ = span/3``."""
        level = float(values[int(np.argmax(x))])
        fall = float(values[int(np.argmin(x))]) - level
        return (np.array([level, fall, _span(x) / 3.0]),)


@dataclass(frozen=True, slots=True)
class BinaryExponentialForm:
    """The binary exponential ``μ − A·2^(γx)``; ``γ`` may stop on a bound of its range."""

    name = "binary_exponential"
    parameters = ("mu", "A", "gamma")
    held_candidates = (2,)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``μ − A·2^(γx)``."""
        return params[0] - params[1] * 2.0 ** (params[2] * x)

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(μ, A, γ) = (1, −2^(γx), −A·ln 2·x·2^(γx))``."""
        power = 2.0 ** (params[2] * x)
        return np.column_stack([np.ones_like(x), -power, -params[1] * math.log(2.0) * x * power])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``μ``: the loss vanishes at ``x → −∞`` for ``γ > 0``."""
        return float(params[0])

    def formula(self, symbol: str) -> str:
        """Return ``μ − A·2^(γ·x)`` with the symbol of the abscissa."""
        return f"μ − A·2^(γ·{symbol})"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A ≥ 0`` and ``γ`` in ``GAMMA_RANGE``."""
        return np.array([-np.inf, 0.0, GAMMA_RANGE[0]]), np.array([np.inf, np.inf, GAMMA_RANGE[1]])

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``μ`` at the largest value, ``A = 1``, ``γ = 0.5``."""
        return (np.array([float(values.max()), 1.0, 0.5]),)


@dataclass(frozen=True, slots=True)
class SaturationForm:
    """The saturation ``μ − A·(1 − e^(−x/x_c))``; ``x_c`` may stop on a bound of its range."""

    name = "saturation"
    parameters = ("mu", "A", "x_c")
    held_candidates = (2,)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``μ − A·(1 − e^(−x/x_c))``."""
        return params[0] - params[1] * (1.0 - np.exp(-x / params[2]))

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(μ, A, x_c) = (1, e^(−x/x_c) − 1, A·x·e^(−x/x_c)/x_c²)``."""
        rest = np.exp(-x / params[2])
        return np.column_stack(
            [np.ones_like(x), rest - 1.0, params[1] * x * rest / (params[2] * params[2])]
        )

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the law is over a load in its own units, its shift is at ``x = 0``."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``μ − A·(1 − e^(−x/x_c))`` with the symbol of the abscissa."""
        return f"μ − A·(1 − e^(−{symbol}/{symbol}_c))"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A ≥ 0`` and ``x_c`` from a thousandth to 1000 spans of the abscissae."""
        return _length_range(x)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``μ`` and ``A`` from the ends of the axis, ``x_c = span/2``."""
        level = float(values[int(np.argmin(x))])
        fall = level - float(values[int(np.argmax(x))])
        return (np.array([level, fall, 0.5 * _span(x)]),)


@dataclass(frozen=True, slots=True)
class PowerForm:
    """The power law ``λ − A·(x^γ − 1)`` over a load from one; ``γ`` may stop on a bound."""

    name = "power"
    parameters = ("lambda", "A", "gamma")
    held_candidates = (2,)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``λ − A·(x^γ − 1)``."""
        return params[0] - params[1] * (x ** params[2] - 1.0)

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(λ, A, γ) = (1, 1 − x^γ, −A·x^γ·ln x)``."""
        grown = x ** params[2]
        return np.column_stack([np.ones_like(x), 1.0 - grown, -params[1] * grown * np.log(x)])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the law is over a load from one, its shift is at ``x = 1``."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``λ − A·(x^γ − 1)`` with the symbol of the abscissa."""
        return f"λ − A·({symbol}^γ − 1)"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A ≥ 0`` and ``γ`` in ``POWER_RANGE``."""
        return np.array([-np.inf, 0.0, POWER_RANGE[0]]), np.array([np.inf, np.inf, POWER_RANGE[1]])

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``γ = 1``, ``λ`` and ``A`` of the line through the ends."""
        level = float(values[int(np.argmin(x))])
        fall = level - float(values[int(np.argmax(x))])
        return (np.array([level, fall / max(float(x.max()) - 1.0, 1.0), 1.0]),)
