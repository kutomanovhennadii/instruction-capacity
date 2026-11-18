"""The law of the length slices of a grid: the adapter of a form, the fit, the choice, the record."""

from __future__ import annotations

import numpy as np
import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.forms.growth_forms import BinaryExponentialForm
from iec.passport.laws.forms.hyperbolic_forms import AsymptoticHyperbolicForm, SlantedHyperbolicForm
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.laws.grid_slices import SlicedForm, choose_slice_form, slice_entry, slice_value

ELL = np.tile(np.linspace(0.0, 6.0, 7), 3)
SLOTS = np.repeat(np.arange(3), 7).astype(float)
X = np.column_stack([ELL, SLOTS])
LEVELS = np.array([10.0, 8.0, 6.0])
FORMS = (BinaryExponentialForm(), AsymptoticHyperbolicForm(), SlantedHyperbolicForm())


def _exponent_cells() -> np.ndarray:
    """Return cells exactly on binary exponentials A = 0.5, γ = 0.4 with the levels of the slices."""
    return np.array(
        [
            slice_value(BinaryExponentialForm(), np.array([0.5, 0.4]), LEVELS[int(slot)], np.array([ell]))[0]
            for ell, slot in X
        ]
    )


def test_the_adapter_shares_the_curve_and_owns_the_levels() -> None:
    sliced = SlicedForm(BinaryExponentialForm(), 3)
    assert sliced.parameters == ("A", "gamma", "mu_0", "mu_1", "mu_2")
    assert sliced.held_candidates == (1,)
    lower, upper = sliced.bounds(X)
    assert lower.tolist()[:2] == [-np.inf, 0.005]
    assert upper.tolist()[:2] == [np.inf, 2.0]
    assert len(sliced.starts(X, _exponent_cells())) == 2


def test_a_form_without_a_level_cannot_be_the_law_of_slices() -> None:
    with pytest.raises(ContractViolationError):
        SlicedForm(LinearForm(), 3)


def test_the_slices_take_the_form_they_were_drawn_from() -> None:
    law = choose_slice_form(FORMS, X, _exponent_cells(), np.full(len(X), 1e-4), 3)
    assert not isinstance(law, str)
    assert law.form.name == "binary_exponential"
    assert law.shared == pytest.approx([0.5, 0.4], abs=1e-6)
    assert law.levels == pytest.approx(LEVELS, abs=1e-6)
    assert law.level_shares(X) == pytest.approx(np.ones(len(X)))


def test_the_record_of_the_slices_names_the_form_and_the_shared_parameters() -> None:
    law = choose_slice_form(FORMS, X, _exponent_cells(), np.full(len(X), 1e-4), 3)
    assert not isinstance(law, str)
    rows = law.rows(list(range(len(X))), len(X))[: law.width]
    entry = slice_entry(law, rows, np.eye(len(X) + 1) * 1e-4, "ℓ")
    assert entry["law_form"] == "binary_exponential"
    assert entry["established_form"] == "binary_exponential"
    assert set(entry["params"]) == {"A", "gamma"}
    assert entry["n_points"] == len(X) and entry["n_parameters"] == 5
    assert [item["form"] for item in entry["candidates"]] == [form.name for form in FORMS]
