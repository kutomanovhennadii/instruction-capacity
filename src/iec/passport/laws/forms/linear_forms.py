"""Forms linear in their parameters: the line, the logarithmic law, the logarithmic law with a line.

The line ``a + b·x`` goes over an abscissa that is already a logarithm - the levels of the length
slices over ``log₂(v/20)``. The logarithmic law ``a + b·log₂(x/x₀)`` goes over a load in its own
units - the number of rules, the length of a text - with the reference ``x₀`` where the logarithm
is zero. The logarithmic law with a line ``a + b·log₂(x/x₀) + c·x`` adds a term linear in the load:
over the density of replacements it lets the law rise at rare replacements and have a hill. None of
them has a limit at ``x → −∞``: their shift is the value at the zero of the axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.law_form import LAW_FORM_INVALID, unbounded_range


def _check_reference(reference: float) -> None:
    """Check the reference of a logarithmic form.

    Raises:
        ConfigError: ``LAW_FORM_INVALID`` - the reference is not positive.
    """
    if not reference > 0.0:
        raise ConfigError(
            f"the reference of a logarithmic form must be positive, got {reference}",
            error_code=LAW_FORM_INVALID,
        )


def _logarithm_text(symbol: str, reference: float) -> str:
    """Return ``log₂ x`` for the reference one, otherwise ``log₂(x/x₀)`` with the number ``x₀``."""
    if reference == 1.0:
        return f"log₂ {symbol}"
    return f"log₂({symbol}/{reference:.10g})"


@dataclass(frozen=True, slots=True)
class LinearForm:
    """The line ``a + b·x`` over the abscissa the block gives."""

    name = "linear"
    parameters = ("a", "b")
    held_candidates = ()

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``a + b·x``."""
        return params[0] + params[1] * x

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(a, b) = (1, x)``."""
        return np.column_stack([np.ones_like(x), x])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: a line has no horizontal asymptote."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``a + b·x`` with the symbol of the abscissa."""
        return f"a + b·{symbol}"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return no bounds."""
        return unbounded_range(2)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start at zero: a fit of a line converges from any start."""
        return (np.zeros(2),)


@dataclass(frozen=True, slots=True)
class LogarithmicForm:
    """The logarithmic law ``a + b·log₂(x/x₀)``.

    ``reference`` - the load ``x₀`` at which the logarithm is zero: one rule for the number of
    rules, 20 words for the length of a text; with the reference one the law is ``a + b·log₂ x``.
    """

    reference: float = 1.0
    name = "logarithmic"
    parameters = ("a", "b")
    held_candidates = ()

    def __post_init__(self) -> None:
        """Check the reference.

        Raises:
            ConfigError: ``LAW_FORM_INVALID`` - the reference is not positive.
        """
        _check_reference(self.reference)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``a + b·log₂(x/x₀)``."""
        return params[0] + params[1] * np.log2(x / self.reference)

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(a, b) = (1, log₂(x/x₀))``."""
        return np.column_stack([np.ones_like(x), np.log2(x / self.reference)])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: a logarithmic law has no horizontal asymptote."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``a + b·log₂(x/x₀)`` with the symbol of the abscissa and the number ``x₀``."""
        return f"a + b·{_logarithm_text(symbol, self.reference)}"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return no bounds."""
        return unbounded_range(2)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start at zero: the law is linear in its parameters."""
        return (np.zeros(2),)


@dataclass(frozen=True, slots=True)
class LogLinearForm:
    """The logarithmic law with a line ``a + b·log₂(x/x₀) + c·x``.

    ``reference`` - the load ``x₀`` at which the logarithm is zero; the linear term goes over the
    load itself.
    """

    reference: float = 1.0
    name = "log_linear"
    parameters = ("a", "b", "c")
    held_candidates = ()

    def __post_init__(self) -> None:
        """Check the reference.

        Raises:
            ConfigError: ``LAW_FORM_INVALID`` - the reference is not positive.
        """
        _check_reference(self.reference)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``a + b·log₂(x/x₀) + c·x``."""
        return params[0] + params[1] * np.log2(x / self.reference) + params[2] * x

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(a, b, c) = (1, log₂(x/x₀), x)``."""
        return np.column_stack([np.ones_like(x), np.log2(x / self.reference), x])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the law has no horizontal asymptote."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``a + b·log₂(x/x₀) + c·x`` with the symbol of the abscissa and ``x₀``."""
        return f"a + b·{_logarithm_text(symbol, self.reference)} + c·{symbol}"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return no bounds."""
        return unbounded_range(3)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start at zero: the law is linear in its parameters."""
        return (np.zeros(3),)


@dataclass(frozen=True, slots=True)
class LogLinearLoadForm:
    """The logarithmic law with a line over the logarithm of a load: ``a + b·x + c·x₀·2^x``.

    ``x = log₂(p/x₀)`` is the abscissa of the block and ``x₀·2^x`` the load itself, so this is the
    logarithmic law with a line ``a + b·log₂(p/x₀) + c·p`` written over the abscissa of its block: over
    the density of F it lets the law rise at rare replacements and have a hill, which the monotone
    candidates of the block do not describe. ``reference`` - the load ``x₀`` at ``x = 0``.
    """

    reference: float = 1.0
    name = "log_linear_load"
    parameters = ("a", "b", "c")
    held_candidates = ()

    def __post_init__(self) -> None:
        """Check the reference.

        Raises:
            ConfigError: ``LAW_FORM_INVALID`` - the reference is not positive.
        """
        _check_reference(self.reference)

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``a + b·x + c·x₀·2^x``."""
        return params[0] + params[1] * x + params[2] * self.reference * 2.0**x

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``∂/∂(a, b, c) = (1, x, x₀·2^x)``."""
        return np.column_stack([np.ones_like(x), x, self.reference * 2.0**x])

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the law has no horizontal asymptote."""
        return None

    def formula(self, symbol: str) -> str:
        """Return ``a + b·x + c·x₀·2^x`` with the symbol of the abscissa and the number ``x₀``."""
        return f"a + b·{symbol} + c·{self.reference:.10g}·2^({symbol})"

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return no bounds."""
        return unbounded_range(3)

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one start at zero: the law is linear in its parameters."""
        return (np.zeros(3),)
