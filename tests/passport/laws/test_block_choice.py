"""The law of a block: the established form, the threshold, the outliers, the line at last."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.laws.block_choice import (
    REASON_FEW_POINTS,
    BlockChoice,
    LawStack,
    block_law,
    block_shift,
    choose_law,
    law_entry,
    unit_rows,
)
from iec.passport.laws.forms.growth_forms import BinaryExponentialForm
from iec.passport.laws.forms.hyperbolic_forms import HyperbolicForm, SlantedHyperbolicForm
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.laws.point_weights import variance_weights

#: The forms of a block: the line established, the hyperbolas and the exponential as alternatives.
FORMS = (LinearForm(), HyperbolicForm(), SlantedHyperbolicForm(), BinaryExponentialForm())
X = np.linspace(0.5, 9.5, 10)
WEIGH = variance_weights(np.full(10, 1e-4))
#: A small even scatter, so that the residuals of a fit are not rounding noise.
SCATTER = np.array([0.01, -0.012, 0.008, -0.01, 0.011, -0.009, 0.012, -0.008, 0.009, -0.011])


def _chosen(values: np.ndarray) -> BlockChoice:
    choice = choose_law(FORMS, X, values, WEIGH)
    assert isinstance(choice, BlockChoice)
    return choice


def test_the_established_form_stays_on_data_it_describes() -> None:
    choice = _chosen(2.0 - 0.5 * X + SCATTER)
    assert choice.form.name == "linear"
    assert choice.record()["established_form"] == "linear"
    assert choice.kept is not None and choice.kept.all()


def test_an_alternative_replaces_the_established_form_beyond_the_threshold() -> None:
    hyperbola = SlantedHyperbolicForm().value(np.array([3.0, -1.5, 4.0, 1.0]), X) + SCATTER
    choice = _chosen(hyperbola)
    assert choice.form.name == "slanted_hyperbolic"
    assert choice.record()["established_form"] == "linear"


def test_every_form_is_recorded_with_its_quasi_criterion() -> None:
    choice = _chosen(2.0 - 0.5 * X + SCATTER)
    assert [entry["form"] for entry in choice.candidates] == [form.name for form in FORMS]
    line = choice.candidates[0]
    assert line["criterion"] == pytest.approx(line["chi2"] / choice.c_hat + 2.0 * line["free"])
    assert choice.c_hat >= 1.0
    assert choice.record()["law_form"] == "linear"


def test_an_outlier_leaves_the_law_and_is_written() -> None:
    values = 2.0 - 0.5 * X + SCATTER
    values[6] += 5.0
    choice = _chosen(values)
    assert choice.form.name == "linear"
    assert choice.kept is not None
    assert np.flatnonzero(~choice.kept).tolist() == [6]
    law = block_law(choice, unit_rows(list(range(10)), 10), [f"b{k}" for k in range(10)], X, False)
    assert law.outliers == ("b6",)
    entry = law_entry(law, np.eye(11) * 1e-4, "x")
    assert entry["outliers"] == ["b6"] and entry["n_points"] == 9


def test_a_form_with_as_many_parameters_as_bins_does_not_compete() -> None:
    choice = choose_law(FORMS, X[:4], 2.0 - 0.5 * X[:4] + SCATTER[:4], variance_weights(np.full(4, 1e-4)))
    assert isinstance(choice, BlockChoice)
    hyperbola = next(entry for entry in choice.candidates if entry["form"] == "slanted_hyperbolic")
    assert hyperbola == {
        "form": "slanted_hyperbolic",
        "converged": False,
        "reason": REASON_FEW_POINTS,
    }


def test_two_bins_take_the_line_and_one_bin_has_no_law() -> None:
    two = choose_law(FORMS, X[:2], np.array([1.0, 0.5]), variance_weights(np.full(2, 1e-4)))
    assert isinstance(two, BlockChoice)
    assert two.fallback and two.form.name == "linear"
    assert two.record()["fallback"] == "line"
    assert (
        choose_law(FORMS, X[:1], np.array([1.0]), variance_weights(np.ones(1)))
        == REASON_FEW_POINTS
    )


def test_the_shift_is_the_asymptote_or_the_law_at_zero() -> None:
    hyperbola = np.array([3.0, -1.5, 4.0, 1.0])
    assert block_shift(SlantedHyperbolicForm(), hyperbola, True) == pytest.approx(3.0)
    assert block_shift(SlantedHyperbolicForm(), hyperbola, False) == pytest.approx(
        float(SlantedHyperbolicForm().value(hyperbola, np.array([0.0]))[0])
    )
    # A line has no asymptote: its shift is the law at the reference of the abscissa.
    assert block_shift(LinearForm(), np.array([2.0, -0.5]), True) == pytest.approx(2.0)
    # A hyperbola over the share of the answer has no value at zero: the smallest abscissa.
    assert block_shift(HyperbolicForm(), np.array([1.0, 0.1]), False, 0.05) == pytest.approx(3.0)


def test_the_rows_of_a_law_carry_its_sensitivity_to_the_coefficients() -> None:
    choice = _chosen(2.0 - 0.5 * X + SCATTER)
    columns = [3, 1, 4, 0, 5, 9, 2, 6, 8, 7]
    law = block_law(choice, unit_rows(columns, 10), [f"b{k}" for k in range(10)], X, False)
    assert law.rows[:, columns] == pytest.approx(choice.fit.sensitivity)
    stack = LawStack({"law": (law.params, law.rows)})
    gradient = stack.gradient(lambda vector: law.shift(stack.part(vector, "law")))
    assert gradient == pytest.approx(law.rows[0], abs=1e-8)
    entry = law_entry(law, np.eye(11) * 1e-4, "x")
    assert entry["law_form"] == "linear"
    assert set(entry["params"]) == {"a", "b"}
    assert entry["half"]["a"] > 0.0
    assert entry["n_points"] == 10 and entry["outliers"] == []
