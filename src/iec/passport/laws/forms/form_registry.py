"""The registry of the forms of laws: the name of a form and the way to build it.

A form is built by its name, with the options it accepts: the logarithmic forms take the reference
``x₀`` of their axis, the others nothing. A new form is a new class of its module and a line of
``LAW_FORMS`` (specification §2.9): neither the fit nor the assignment changes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.growth_forms import (
    BinaryExponentialForm,
    ExponentialForm,
    PowerForm,
    SaturationForm,
)
from iec.passport.laws.forms.hyperbolic_forms import (
    AsymptoticHyperbolicForm,
    FractionalHyperbolicForm,
    HyperbolicForm,
    SlantedHyperbolicForm,
    TwoSlopeHyperbolicForm,
)
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.forms.linear_forms import (
    LinearForm,
    LogarithmicForm,
    LogLinearForm,
    LogLinearLoadForm,
)

#: Error code of a name that the registry does not know.
LAW_FORM_UNKNOWN = "LAW_FORM_UNKNOWN"
#: Error code of an option that the form does not take.
LAW_FORM_OPTIONS_INVALID = "LAW_FORM_OPTIONS_INVALID"


@dataclass(frozen=True, slots=True)
class FormSpec:
    """How the registry builds a form: its factory and the options the factory takes."""

    factory: Callable[..., LawForm]
    options: frozenset[str] = field(default_factory=frozenset)


#: Every form of the collection by its name.
LAW_FORMS: Mapping[str, FormSpec] = {
    "linear": FormSpec(LinearForm),
    "logarithmic": FormSpec(LogarithmicForm, frozenset({"reference"})),
    "log_linear": FormSpec(LogLinearForm, frozenset({"reference"})),
    "log_linear_load": FormSpec(LogLinearLoadForm, frozenset({"reference"})),
    "hyperbolic": FormSpec(HyperbolicForm),
    "fractional_hyperbolic": FormSpec(FractionalHyperbolicForm),
    "slanted_hyperbolic": FormSpec(SlantedHyperbolicForm),
    "two_slope_hyperbolic": FormSpec(TwoSlopeHyperbolicForm),
    "asymptotic_hyperbolic": FormSpec(AsymptoticHyperbolicForm),
    "exponential": FormSpec(ExponentialForm),
    "binary_exponential": FormSpec(BinaryExponentialForm),
    "saturation": FormSpec(SaturationForm),
    "power": FormSpec(PowerForm),
}


def build_form(name: str, options: Mapping[str, float] | None = None) -> LawForm:
    """Build a form by its name with its options.

    Args:
        name: The name of the form in ``LAW_FORMS``.
        options: The options of the form, such as ``{"reference": 20.0}``; ``None`` - none.

    Returns:
        The form.

    Raises:
        ConfigError: ``LAW_FORM_UNKNOWN`` - the registry has no such form;
            ``LAW_FORM_OPTIONS_INVALID`` - an option that the form does not take;
            ``LAW_FORM_INVALID`` - a value of an option that the form rejects.
    """
    spec = LAW_FORMS.get(name)
    if spec is None:
        known = ", ".join(sorted(LAW_FORMS))
        raise ConfigError(
            f"unknown form of a law '{name}'; known: {known}", error_code=LAW_FORM_UNKNOWN
        )
    given = dict(options or {})
    extra = sorted(set(given) - spec.options)
    if extra:
        raise ConfigError(
            f"the form '{name}' takes no options {extra}", error_code=LAW_FORM_OPTIONS_INVALID
        )
    return spec.factory(**given)
