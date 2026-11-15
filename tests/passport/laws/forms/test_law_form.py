"""The interface of a form: the range without bounds and the shift of a law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.forms.growth_forms import ExponentialForm
from iec.passport.laws.forms.hyperbolic_forms import SlantedHyperbolicForm
from iec.passport.laws.forms.law_form import LAW_SHIFT_UNDEFINED, law_shift, unbounded_range
from iec.passport.laws.forms.linear_forms import LogarithmicForm


def test_a_range_without_bounds_is_infinite_on_both_sides() -> None:
    lower, upper = unbounded_range(3)
    assert lower.tolist() == [-math.inf] * 3
    assert upper.tolist() == [math.inf] * 3


def test_the_shift_at_minus_infinity_is_the_limit_of_the_form() -> None:
    params = np.array([18.5, -11.0, 10.0, 4.8])
    assert law_shift(SlantedHyperbolicForm(), params, -math.inf) == 18.5


def test_the_shift_at_a_number_is_the_value_of_the_law_there() -> None:
    params = np.array([-0.2, 1.1, 0.55])
    assert law_shift(ExponentialForm(), params, 0.0) == -0.2 + 1.1


def test_a_form_without_a_limit_has_no_shift_at_minus_infinity() -> None:
    with pytest.raises(ContractViolationError) as error:
        law_shift(LogarithmicForm(), np.array([1.0, -0.5]), -math.inf)
    assert error.value.error_code == LAW_SHIFT_UNDEFINED
