"""The forms linear in their parameters: the line and the two logarithmic laws."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.law_form import LAW_FORM_INVALID, LawForm
from iec.passport.laws.forms.linear_forms import LinearForm, LogarithmicForm, LogLinearForm

FORMS = (LinearForm(), LogarithmicForm(20.0), LogLinearForm(0.001))


def test_the_line_is_a_plus_b_x() -> None:
    values = LinearForm().value(np.array([1.0, 2.0]), np.array([0.0, 1.0, 3.0]))
    assert values.tolist() == [1.0, 3.0, 7.0]


def test_the_logarithmic_law_is_a_at_its_reference() -> None:
    values = LogarithmicForm(20.0).value(np.array([3.0, -0.5]), np.array([20.0, 40.0, 80.0]))
    assert values.tolist() == [3.0, 2.5, 2.0]


def test_the_logarithmic_law_with_a_line_adds_c_x() -> None:
    values = LogLinearForm(0.5).value(np.array([1.0, 2.0, -3.0]), np.array([0.5, 1.0]))
    assert values.tolist() == [-0.5, 0.0]


def test_the_reference_one_gives_the_plain_logarithm_bit_for_bit() -> None:
    params = np.array([2.6271921234703326, 0.2198052177487886])
    x = np.array([0.0011, 0.37, 3.0, 110.6, 4100.0])
    plain = params[0] + params[1] * np.log2(x)
    assert LogarithmicForm().value(params, x).tobytes() == plain.tobytes()


def test_the_formulas_name_the_symbol_and_the_reference() -> None:
    assert LinearForm().formula("u") == "a + b·u"
    assert LogarithmicForm().formula("v") == "a + b·log₂ v"
    assert LogarithmicForm(20.0).formula("v") == "a + b·log₂(v/20)"
    assert LogLinearForm(0.0009765625).formula("p") == "a + b·log₂(p/0.0009765625) + c·p"


@pytest.mark.parametrize("build", [LogarithmicForm, LogLinearForm])
@pytest.mark.parametrize("reference", [0.0, -1.0, math.nan])
def test_a_reference_that_is_not_positive_is_rejected(
    build: Callable[[float], LawForm], reference: float
) -> None:
    with pytest.raises(ConfigError) as error:
        build(reference)
    assert error.value.error_code == LAW_FORM_INVALID


@pytest.mark.parametrize("form", FORMS)
def test_the_linear_forms_have_no_limit_no_bounds_and_nothing_held(form: LawForm) -> None:
    count = len(form.parameters)
    lower, upper = form.bounds(np.array([1.0, 2.0]))
    assert form.limit(np.zeros(count)) is None
    assert lower.tolist() == [-math.inf] * count
    assert upper.tolist() == [math.inf] * count
    assert form.held_candidates == ()


@pytest.mark.parametrize("form", FORMS)
def test_the_linear_forms_start_at_zero(form: LawForm) -> None:
    starts = form.starts(np.array([1.0, 2.0]), np.array([0.5, 0.25]))
    assert [start.tolist() for start in starts] == [[0.0] * len(form.parameters)]


def test_the_names_of_the_parameters() -> None:
    assert LinearForm().parameters == ("a", "b")
    assert LogarithmicForm().parameters == ("a", "b")
    assert LogLinearForm().parameters == ("a", "b", "c")


@pytest.mark.parametrize(
    ("form", "params", "x"),
    [
        (LinearForm(), [0.3, -1.2], [0.0, 1.0, 2.5]),
        (LogarithmicForm(20.0), [2.0, 0.7], [10.0, 20.0, 80.0]),
        (LogLinearForm(0.001), [1.0, 0.2, -3.0], [0.001, 0.01, 0.5]),
    ],
)
def test_the_derivatives_are_those_of_the_value(
    numeric_jacobian: Callable[..., np.ndarray],
    form: LawForm,
    params: list[float],
    x: list[float],
) -> None:
    point, abscissae = np.array(params), np.array(x)
    np.testing.assert_allclose(
        form.jacobian(point, abscissae),
        numeric_jacobian(form, point, abscissae),
        rtol=1e-6,
        atol=1e-7,
    )
