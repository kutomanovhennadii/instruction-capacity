"""The registry of the forms: every form by its name, its options and the errors of a name."""

from __future__ import annotations

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.form_registry import (
    LAW_FORM_OPTIONS_INVALID,
    LAW_FORM_UNKNOWN,
    LAW_FORMS,
    build_form,
)
from iec.passport.laws.forms.law_form import LAW_FORM_INVALID
from iec.passport.laws.forms.linear_forms import LogarithmicForm

NAMES = {
    "linear",
    "logarithmic",
    "log_linear",
    "hyperbolic",
    "fractional_hyperbolic",
    "slanted_hyperbolic",
    "two_slope_hyperbolic",
    "asymptotic_hyperbolic",
    "exponential",
    "binary_exponential",
    "saturation",
    "power",
    "log_linear_load",
}


def test_the_registry_holds_the_thirteen_forms_of_the_collection() -> None:
    assert set(LAW_FORMS) == NAMES


@pytest.mark.parametrize("name", sorted(NAMES))
def test_every_form_is_built_under_its_own_name(name: str) -> None:
    assert build_form(name).name == name


def test_a_logarithmic_form_takes_its_reference() -> None:
    form = build_form("logarithmic", {"reference": 20.0})
    assert form == LogarithmicForm(20.0)
    assert form.formula("v") == "a + b·log₂(v/20)"


def test_an_unknown_form_is_a_configuration_error_naming_the_known_ones() -> None:
    with pytest.raises(ConfigError) as error:
        build_form("cubic")
    assert error.value.error_code == LAW_FORM_UNKNOWN
    assert "slanted_hyperbolic" in error.value.message


def test_an_option_the_form_does_not_take_is_rejected() -> None:
    with pytest.raises(ConfigError) as error:
        build_form("linear", {"reference": 2.0})
    assert error.value.error_code == LAW_FORM_OPTIONS_INVALID


def test_a_value_the_form_rejects_is_a_configuration_error() -> None:
    with pytest.raises(ConfigError) as error:
        build_form("log_linear", {"reference": 0.0})
    assert error.value.error_code == LAW_FORM_INVALID
