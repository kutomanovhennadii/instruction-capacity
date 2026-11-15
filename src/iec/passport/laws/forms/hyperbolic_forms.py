"""Hyperbolic forms: the hyperbola, the fractional, the slanted, the two-slope and the two-asymptote hyperbola.

The hyperbola ``c + A/x`` falls from infinity at the vanishing load to ``c``; it is the
approximation of the variance whose fractional hyperbola has degenerated. The fractional hyperbola
``c + A/(1 + x/x₀)`` falls from ``c + A`` at ``x = 0`` to ``c``, half of the fall at ``x = x₀``.
The slanted hyperbola ``μ + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]`` tends to the horizontal
asymptote ``μ`` for ``x ≪ x*`` and to the slanted asymptote ``μ + κ·(x − x*)`` for ``x ≫ x*``;
``x*`` is where the asymptotes cross, ``ω`` the width of the corner. The two-slope hyperbola
``μ + β·(x − x*) + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]`` stands on two slanted asymptotes, the
slope ``β`` on the left and ``β + κ`` on the right, of any sign: it can fall from its top to both
sides, as a hill, or rise at one end, and at ``β = 0`` it is the slanted hyperbola. The
two-asymptote hyperbola ``[(κx + λ) + μ − √((κx + λ − μ)² + κ²ω²)]/2`` is the smooth minimum of
the level ``μ`` and the line ``κx + λ``: the same slanted hyperbola written by the slope and the
intercept of its slanted asymptote, ``x* = (μ − λ)/κ``, as a length slice of the grid of P_sync is
written. The two slanted forms have the limit ``μ`` at ``x → −∞`` for a falling slanted asymptote,
``κ < 0``; the two-slope hyperbola has no horizontal asymptote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iec.passport.laws.forms.law_form import unbounded_range

#: The smallest width of a corner, and half the smallest top of an axis of a slanted hyperbola.
WIDTH_FLOOR = 0.001
#: The range of ``x₀`` of a fractional hyperbola as shares of the smallest and the largest abscissa.
X0_FLOOR_SHARE = 1e-3
X0_CEILING_SHARE = 1e3


@dataclass(frozen=True, slots=True)
class HyperbolicForm:
    """The hyperbola ``c + A/x``."""

    name = "hyperbolic"
    parameters = ("c", "A")
    held_candidates = ()

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``c + A/x``."""
        return params[0] + params[1] / x

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(c, A) = (1, 1/x)``."""
        return np.column_stack([np.ones_like(x), 1.0 / x])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the hyperbola is a law over a load in its own units."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``c + A/x`` with the symbol of the abscissa."""
        return f"c + A/{symbol}"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return no bounds."""
        return unbounded_range(2)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``c`` at the largest abscissa, the fall to the smallest in ``A``."""
        level = float(values[int(np.argmax(x))])
        return (np.array([level, (float(values[int(np.argmin(x))]) - level) * float(x.min())]),)


@dataclass(frozen=True, slots=True)
class FractionalHyperbolicForm:
    """The fractional hyperbola ``c + A/(1 + x/x₀)``; ``x₀`` may stop on its lower bound."""

    name = "fractional_hyperbolic"
    parameters = ("c", "A", "x0")
    held_candidates = (2,)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``c + A/(1 + x/x₀)``."""
        return params[0] + params[1] / (1.0 + x / params[2])

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(c, A, x₀) = (1, 1/q, A·x/(x₀²·q²))``, ``q = 1 + x/x₀``."""
        q = 1.0 + x / params[2]
        return np.column_stack(
            [np.ones_like(x), 1.0 / q, params[1] * x / (params[2] * params[2] * q * q)]
        )

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the fractional hyperbola is a law over a load in its own units."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``c + A/(1 + x/x₀)`` with the symbol of the abscissa."""
        return f"c + A/(1 + {symbol}/{symbol}₀)"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``x₀`` from a thousandth of the smallest to 1000 of the largest ``x``."""
        lower = np.array([-np.inf, -np.inf, X0_FLOOR_SHARE * float(x.min())])
        upper = np.array([np.inf, np.inf, X0_CEILING_SHARE * float(x.max())])
        return lower, upper

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``c`` and ``A`` from the ends of the axis, ``x₀`` at its start."""
        level = float(values[int(np.argmax(x))])
        fall = float(values[int(np.argmin(x))]) - level
        return (np.array([level, fall, float(x.min())]),)


@dataclass(frozen=True, slots=True)
class SlantedHyperbolicForm:
    """The slanted hyperbola ``μ + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]``.

    The range: a falling slanted asymptote ``κ ≤ 0``, the corner ``x*`` and its width ``ω``
    inside the axis; the corner and the width may stop on a bound of the range.
    """

    name = "slanted_hyperbolic"
    parameters = ("mu", "kappa", "corner", "omega")
    held_candidates = (2, 3)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``μ + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]``."""
        mu, kappa, corner, omega = params
        t = x - corner
        return mu + 0.5 * kappa * (t + np.sqrt(t * t + omega * omega))

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(μ, κ, x*, ω) = (1, (t + R)/2, −(κ/2)·(1 + t/R), (κ/2)·ω/R)``.

        ``t = x − x*`` and ``R = √(t² + ω²)``; ``R ≥ ω > 0`` because the width has a positive bound.
        """
        kappa, corner, omega = params[1], params[2], params[3]
        t = x - corner
        r = np.sqrt(t * t + omega * omega)
        return np.column_stack(
            [np.ones_like(t), 0.5 * (t + r), -0.5 * kappa * (1.0 + t / r), 0.5 * kappa * omega / r]
        )

    def limit(self, params: np.ndarray) -> float | None:
        """Return the horizontal asymptote ``μ``."""
        return float(params[0])

    def formula(self, symbol: str) -> str:
        """Return the formula with the symbol of the abscissa."""
        return f"μ + (κ/2)·[({symbol} − {symbol}*) + √(({symbol} − {symbol}*)² + ω²)]"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``κ ≤ 0``, ``0 ≤ x* ≤ top``, ``0.001 ≤ ω ≤ top``; ``top`` - the largest ``x``."""
        top = _top(x)
        return np.array([-np.inf, -np.inf, 0.0, WIDTH_FLOOR]), np.array([np.inf, 0.0, top, top])

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return six starts: the level at the smallest abscissa, slopes −1 and −3, three corners.

        The corners stand at a quarter, a half and three quarters of ``top``; the width is 0.5.
        These are the starts of the density law of P_sync.
        """
        top = _top(x)
        level = float(values[int(np.argmin(x))])
        return tuple(
            np.array([level, slope, corner, 0.5])
            for slope in (-1.0, -3.0)
            for corner in (0.25 * top, 0.5 * top, 0.75 * top)
        )


@dataclass(frozen=True, slots=True)
class TwoSlopeHyperbolicForm:
    """The two-slope hyperbola ``μ + β·(x − x*) + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]``.

    The asymptotes have the slopes ``β`` on the left and ``β + κ`` on the right, of any sign. The
    range: the corner ``x*`` and its width ``ω`` inside the axis, as for the slanted hyperbola; the
    corner and the width may stop on a bound of the range.
    """

    name = "two_slope_hyperbolic"
    parameters = ("mu", "kappa", "corner", "omega", "beta")
    held_candidates = (2, 3)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``μ + β·(x − x*) + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]``."""
        mu, kappa, corner, omega, beta = params
        t = x - corner
        return mu + beta * t + 0.5 * kappa * (t + np.sqrt(t * t + omega * omega))

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(μ, κ, x*, ω, β) = (1, (t + R)/2, −β − (κ/2)·(1 + t/R), (κ/2)·ω/R, t)``.

        ``t = x − x*`` and ``R = √(t² + ω²)``; ``R ≥ ω > 0`` because the width has a positive bound.
        """
        kappa, corner, omega, beta = params[1], params[2], params[3], params[4]
        t = x - corner
        r = np.sqrt(t * t + omega * omega)
        return np.column_stack(
            [
                np.ones_like(t),
                0.5 * (t + r),
                -beta - 0.5 * kappa * (1.0 + t / r),
                0.5 * kappa * omega / r,
                t,
            ]
        )

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the left asymptote is slanted, so the law has no horizontal asymptote."""
        return None

    def formula(self, symbol: str) -> str:
        """Return the formula with the symbol of the abscissa."""
        t = f"({symbol} − {symbol}*)"
        return f"μ + β·{t} + (κ/2)·[{t} + √({t}² + ω²)]"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``0 ≤ x* ≤ top`` and ``0.001 ≤ ω ≤ top``, the slopes free; ``top`` - the largest ``x``."""
        top = _top(x)
        lower = np.array([-np.inf, -np.inf, 0.0, WIDTH_FLOOR, -np.inf])
        upper = np.array([np.inf, np.inf, top, top, np.inf])
        return lower, upper

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the starts: two lines at three corners, then those of the slanted hyperbola.

        At a corner at a quarter, a half and three quarters of ``top`` with at least two points on
        each side, the lines of least squares to the left and to the right of it give the level at
        the corner, the slope ``β`` on the left and ``κ`` as the change of the slope; the width is
        0.5. Then come the six starts of the slanted hyperbola with ``β = 0``.
        """
        top = _top(x)
        order = np.argsort(x)
        xs = np.asarray(x, dtype=float)[order]
        ys = np.asarray(values, dtype=float)[order]
        found: list[np.ndarray] = []
        for share in (0.25, 0.5, 0.75):
            corner = share * top
            left, right = xs <= corner, xs >= corner
            if left.sum() >= 2 and right.sum() >= 2:
                left_slope, left_level = np.polyfit(xs[left], ys[left], 1)
                right_slope, _ = np.polyfit(xs[right], ys[right], 1)
                at_corner = left_level + left_slope * corner
                found.append(
                    np.array([at_corner, right_slope - left_slope, corner, 0.5, left_slope])
                )
        first = float(ys[0])
        found.extend(
            np.array([first, slope, share * top, 0.5, 0.0])
            for slope in (-1.0, -3.0)
            for share in (0.25, 0.5, 0.75)
        )
        return tuple(found)


@dataclass(frozen=True, slots=True)
class AsymptoticHyperbolicForm:
    """The two-asymptote hyperbola ``[(κx + λ) + μ − √((κx + λ − μ)² + κ²ω²)]/2``.

    The range: a falling line ``κ ≤ 0`` and the width ``ω`` from 0.001 to the span of the axis.
    """

    name = "asymptotic_hyperbolic"
    parameters = ("kappa", "lambda", "omega", "mu")
    held_candidates = ()

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the smooth minimum of the line ``κx + λ`` and the level ``μ``."""
        kappa, intercept, omega, mu = params
        line = kappa * x + intercept
        gap = line - mu
        return 0.5 * (line + mu - np.sqrt(gap * gap + kappa * kappa * omega * omega))

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(κ, λ, ω, μ)`` with ``d = κx + λ − μ`` and ``Q = √(d² + κ²ω²)``.

        ``∂/∂κ = (x − (d·x + κω²)/Q)/2``, ``∂/∂λ = (1 − d/Q)/2``, ``∂/∂ω = −κ²ω/(2Q)``,
        ``∂/∂μ = (1 + d/Q)/2``.
        """
        kappa, intercept, omega, mu = params
        d = kappa * x + intercept - mu
        q = np.sqrt(d * d + kappa * kappa * omega * omega)
        jacobian = np.zeros((len(x), 4))
        jacobian[:, 0] = 0.5 * (x - (d * x + kappa * omega * omega) / q)
        jacobian[:, 1] = 0.5 * (1.0 - d / q)
        jacobian[:, 2] = -0.5 * kappa * kappa * omega / q
        jacobian[:, 3] = 0.5 * (1.0 + d / q)
        return jacobian

    def limit(self, params: np.ndarray) -> float | None:
        """Return the level ``μ``, the horizontal asymptote of a falling line."""
        return float(params[3])

    def formula(self, symbol: str) -> str:
        """Return the formula with the symbol of the abscissa."""
        return f"[(κ·{symbol} + λ) + μ − √((κ·{symbol} + λ − μ)² + κ²·ω²)]/2"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``κ ≤ 0`` and ``0.001 ≤ ω ≤`` the span of the abscissae."""
        span = max(float(x.max() - x.min()), 2.0 * WIDTH_FLOOR)
        lower = np.array([-np.inf, -np.inf, WIDTH_FLOOR, -np.inf])
        upper = np.array([0.0, np.inf, span, np.inf])
        return lower, upper

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start: ``κ = −2``, ``λ = 0``, ``ω = 1``, the level at the largest value."""
        return (np.array([-2.0, 0.0, 1.0, float(values.max())]),)


def _top(x: np.ndarray) -> float:
    """Return the top of the axis of a slanted hyperbola: the largest abscissa, at least 0.002."""
    return max(float(x.max()), 2.0 * WIDTH_FLOOR)
