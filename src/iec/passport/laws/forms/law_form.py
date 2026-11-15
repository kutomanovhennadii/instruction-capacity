"""The interface of a form of a law of the passport and the shift of a law.

A law of the passport is a smooth curve of a few parameters through the fitted bins of a block:
the coefficients of the bins of an axis, plotted over the load of the axis, lie on it. A form is a
family of such curves - linear, logarithmic, hyperbolic, exponential and others - behind one
interface (specification §2.10): the value of the law, its derivatives by the parameters for the
uncertainties by the delta method, its limit where the load vanishes on an axis in logarithms, the
formula for the document, the names of the parameters, the range and the starts of a fit, the
parameters the data may push to a bound. A form knows nothing of a quantity or a block: the block
gives it the abscissa - the load or its logarithm - and the assignment says which form a block
takes (``law_assignment``).

The shift of a block is the value of its law at the zero of its axis, where the load vanishes. On an
axis in logarithms the zero lies at minus infinity, and the shift is the horizontal asymptote of the
law there; on an axis whose load starts at a number - one rule, the start of the answer - it is the
value of the law at that number.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError

#: Error code of a shift asked at minus infinity of a form without a limit there.
LAW_SHIFT_UNDEFINED = "LAW_SHIFT_UNDEFINED"
#: Error code of a form built with an option it cannot take, such as a reference below zero.
LAW_FORM_INVALID = "LAW_FORM_INVALID"


class LawForm(Protocol):
    """A form of a law: a family of curves of a few parameters over the abscissa of a block."""

    @property
    def name(self) -> str:
        """Return the name of the form in the registry."""
        ...

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the names of the parameters in the order of their vector."""
        ...

    @property
    def held_candidates(self) -> tuple[int, ...]:
        """Return the parameters that the data may push to a bound of their range."""
        ...

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the law at the abscissae ``x``."""
        ...

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the derivatives by the parameters at ``x``: ``(points, parameters)``."""
        ...

    def limit(self, params: np.ndarray) -> float | None:
        """Return the horizontal asymptote of the law at ``x → −∞``, or ``None`` without one."""
        ...

    def formula(self, symbol: str) -> str:
        """Return the formula of the law for the document, with ``symbol`` for the abscissa."""
        ...

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the lower and the upper bound of every parameter of a fit at ``x``."""
        ...

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the starts of a fit through the points ``(x, values)``."""
        ...


def unbounded_range(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the range of ``count`` parameters without bounds."""
    return np.full(count, -np.inf), np.full(count, np.inf)


def law_shift(form: LawForm, params: np.ndarray, zero: float) -> float:
    """Return the shift of a law: its value at the zero of its axis.

    Args:
        form: The form of the law.
        params: Its parameters.
        zero: The abscissa where the load vanishes; ``-inf`` on an axis in logarithms.

    Returns:
        The limit of the law at ``x → −∞`` for ``zero = -inf``, otherwise the law at ``zero``.

    Raises:
        ContractViolationError: ``LAW_SHIFT_UNDEFINED`` - ``zero`` is minus infinity and the form
            has no limit there.
    """
    if math.isinf(zero) and zero < 0.0:
        limit = form.limit(params)
        if limit is None:
            raise ContractViolationError(
                f"the form '{form.name}' has no limit at minus infinity: no shift there",
                error_code=LAW_SHIFT_UNDEFINED,
            )
        return limit
    return float(form.value(params, np.array([zero]))[0])
