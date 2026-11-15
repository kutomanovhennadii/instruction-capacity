"""The drawings of the approximations of the spread: the form, the formula, the curve, the parameters."""

from __future__ import annotations

import numpy as np
import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.document.law_drawings import SPREAD_FORM_UNKNOWN, spread_drawing

X = np.array([0.5, 2.0, 8.0])


def test_a_line_is_a_logarithm_of_the_axis() -> None:
    law = {"form": "line", "params": {"a": 2.2, "b": -0.054}}
    drawing = spread_drawing(law)
    assert drawing.formula("v") == "a + b·log₂ v"
    assert np.array_equal(drawing.curve(law, X), 2.2 + -0.054 * np.log2(X))
    assert drawing.parameter_text(law, "v") == "`a = 2.20`, `b = −0.054`"


def test_a_hyperbola_is_the_fractional_hyperbola() -> None:
    law = {"form": "hyperbola", "params": {"c": 0.063, "A": 18.9, "x0": 0.73}}
    drawing = spread_drawing(law)
    assert drawing.formula("m") == "c + A/(1 + m/m₀)"
    assert np.array_equal(drawing.curve(law, X), 0.063 + 18.9 / (1.0 + X / 0.73))
    assert drawing.parameter_text(law, "m") == "`c = 0.06`, `A = 18.90`, `m₀ = 0.73`"


def test_an_inverse_law_is_the_hyperbola() -> None:
    law = {"form": "inverse", "params": {"c": 9.03, "A": 10.3}}
    drawing = spread_drawing(law)
    assert drawing.formula("m") == "c + A/m"
    assert np.array_equal(drawing.curve(law, X), 9.03 + 10.3 / X)
    assert drawing.parameter_text(law, "m") == "`c = 9.03`, `A = 10.30`"


def test_an_unknown_form_is_refused() -> None:
    with pytest.raises(ContractViolationError) as caught:
        spread_drawing({"form": "parabola", "params": {}})
    assert caught.value.error_code == SPREAD_FORM_UNKNOWN
