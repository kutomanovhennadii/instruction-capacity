"""The forms of the approximations of the spread, shared by the text of the passport and its figures.

The passport publishes an approximation of the variance of the request shift by the name of its form:
``line`` - ``a + b·log₂ x``, ``hyperbola`` - ``c + A/(1 + x/x₀)``, ``inverse`` - ``c + A/x``. A
drawing names the form of the collection behind the published name, the names of its parameters in
the order of the form and how the text writes every parameter. The text takes the formula from the
form, and the figure takes the formula and the curve from it: one formula serves both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.document.number_text import number
from iec.passport.laws.forms.hyperbolic_forms import FractionalHyperbolicForm, HyperbolicForm
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.forms.linear_forms import LogarithmicForm

#: Error code of an approximation of the spread under a name no drawing knows.
SPREAD_FORM_UNKNOWN = "SPREAD_FORM_UNKNOWN"
#: The symbols of the axes of the spread in the formulas of its approximations.
SPREAD_SYMBOLS = {"V": "v", "P": "p", "M": "m"}


@dataclass(frozen=True, slots=True)
class SpreadDrawing:
    """The form of a published approximation of the spread and how the text writes its parameters.

    ``parameters`` - the published names in the order of the form; ``labels`` - the names the text
    gives them, ``{symbol}`` standing for the symbol of the axis; ``digits`` - their digits.
    """

    form: LawForm
    parameters: tuple[str, ...]
    labels: tuple[str, ...]
    digits: tuple[int, ...]

    def params(self, law: Mapping[str, Any]) -> np.ndarray:
        """Return the parameters of a published approximation in the order of the form.

        Args:
            law: The approximation: ``form``, ``params``, ``chi2``, ``n_points``.
        """
        return np.array([float(law["params"][name]) for name in self.parameters])

    def curve(self, law: Mapping[str, Any], x: np.ndarray) -> np.ndarray:
        """Return the approximation at the centres ``x``.

        Args:
            law: The approximation.
            x: The centres of the groups of the axis.
        """
        return self.form.value(self.params(law), x)

    def formula(self, symbol: str) -> str:
        """Return the formula of the approximation with the symbol of the axis.

        Args:
            symbol: The symbol of the axis: ``v``, ``p`` or ``m``.
        """
        return self.form.formula(symbol)

    def parameter_text(self, law: Mapping[str, Any], symbol: str) -> str:
        """Return the parameters of the approximation as the text of the passport writes them.

        Args:
            law: The approximation.
            symbol: The symbol of the axis.
        """
        return ", ".join(
            f"`{label.format(symbol=symbol)} = {number(float(law['params'][name]), digits)}`"
            for name, label, digits in zip(self.parameters, self.labels, self.digits, strict=True)
        )


#: The drawings of the approximations of the spread by their published names.
SPREAD_DRAWINGS: Mapping[str, SpreadDrawing] = {
    "line": SpreadDrawing(LogarithmicForm(), ("a", "b"), ("a", "b"), (2, 3)),
    "hyperbola": SpreadDrawing(
        FractionalHyperbolicForm(), ("c", "A", "x0"), ("c", "A", "{symbol}₀"), (2, 2, 2)
    ),
    "inverse": SpreadDrawing(HyperbolicForm(), ("c", "A"), ("c", "A"), (2, 2)),
}


def spread_drawing(law: Mapping[str, Any]) -> SpreadDrawing:
    """Return the drawing of a published approximation of the spread.

    Args:
        law: The approximation.

    Raises:
        ContractViolationError: ``SPREAD_FORM_UNKNOWN`` - no drawing knows its form.
    """
    drawing = SPREAD_DRAWINGS.get(str(law["form"]))
    if drawing is None:
        raise ContractViolationError(
            f"no drawing of an approximation of the spread '{law['form']}'",
            error_code=SPREAD_FORM_UNKNOWN,
        )
    return drawing
