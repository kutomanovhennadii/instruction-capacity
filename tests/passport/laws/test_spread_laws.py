"""The approximations of the spread by the groups: the laws, the degenerate hyperbola, the forms."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.passport.laws.forms.law_assignment import (
    LAW_FORM_UNSUPPORTED,
    LawAssignmentConfig,
    build_law_assignment,
    load_law_assignment_config,
)
from iec.passport.laws.spread_laws import (
    SPREAD_HYPERBOLA,
    SPREAD_INVERSE,
    SPREAD_LINE,
    spread_forms,
    spread_laws,
)

#: The assignment of the forms of the new folder.
CONFIG = Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml"


def test_the_approximations_of_the_spread_recover_their_laws() -> None:
    centers = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    def rows(law: Any) -> list[dict[str, float]]:
        return [{"center": x, "sigma2": law(x), "half": 0.2 * law(x)} for x in centers]

    spread = {
        "V": rows(lambda x: 3.0 - 0.25 * math.log2(x)),
        "P": rows(lambda x: 1.0 + 0.1 * math.log2(x)),
        "M": rows(lambda x: 0.1 + 12.0 / (1.0 + x / 1.5)),
    }
    laws = spread_laws(spread, {"V": SPREAD_LINE, "P": SPREAD_LINE, "M": SPREAD_HYPERBOLA})
    assert laws["V"]["form"] == SPREAD_LINE
    assert laws["V"]["params"]["a"] == pytest.approx(3.0, abs=1e-6)
    assert laws["V"]["params"]["b"] == pytest.approx(-0.25, abs=1e-6)
    assert laws["P"]["params"]["b"] == pytest.approx(0.1, abs=1e-6)
    assert laws["V"]["chi2"] == pytest.approx(0.0, abs=1e-10)
    assert laws["V"]["n_points"] == len(centers)
    assert laws["M"]["form"] == SPREAD_HYPERBOLA
    assert laws["M"]["params"]["c"] == pytest.approx(0.1, abs=1e-5)
    assert laws["M"]["params"]["A"] == pytest.approx(12.0, rel=1e-5)
    assert laws["M"]["params"]["x0"] == pytest.approx(1.5, rel=1e-5)
    assert laws["M"]["chi2"] == pytest.approx(0.0, abs=1e-8)


def test_an_axis_with_too_few_groups_has_no_approximation() -> None:
    rows = [
        {"center": 1.0, "sigma2": 2.0, "half": 0.4},
        {"center": 8.0, "sigma2": 1.0, "half": 0.2},
    ]
    three = [*rows, {"center": 64.0, "sigma2": 0.5, "half": 0.1}]
    laws = spread_laws({"V": rows, "M": three}, {"V": SPREAD_LINE, "M": SPREAD_HYPERBOLA})
    assert laws == {}


def test_a_hyperbola_on_its_bound_degenerates_into_the_inverse_law() -> None:
    centers = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    rows = [
        {"center": x, "sigma2": 9.0 + 10.0 / x, "half": 0.2 * (9.0 + 10.0 / x)} for x in centers
    ]
    laws = spread_laws({"M": rows}, {"M": SPREAD_HYPERBOLA})
    assert laws["M"]["form"] == SPREAD_INVERSE
    assert set(laws["M"]["params"]) == {"c", "A"}
    assert laws["M"]["params"]["c"] == pytest.approx(9.0, abs=1e-6)
    assert laws["M"]["params"]["A"] == pytest.approx(10.0, abs=1e-6)
    assert laws["M"]["chi2"] == pytest.approx(0.0, abs=1e-10)


def test_a_flat_hyperbola_degenerates_into_the_line() -> None:
    # Constant groups: the hyperbola runs flat, its A and x0 are not determined, the line replaces it.
    centers = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
    rows = [{"center": x, "sigma2": 1.5, "half": 0.3} for x in centers]
    laws = spread_laws({"V": rows}, {"V": SPREAD_HYPERBOLA})
    assert laws["V"]["form"] == SPREAD_LINE
    assert laws["V"]["params"]["a"] == pytest.approx(1.5, abs=1e-6)
    assert laws["V"]["params"]["b"] == pytest.approx(0.0, abs=1e-6)


def test_the_forms_of_the_spread_come_from_the_assignment() -> None:
    assignment = build_law_assignment(load_law_assignment_config(CONFIG))
    assert spread_forms(assignment, "P_sync") == {
        "V": SPREAD_LINE,
        "P": SPREAD_LINE,
        "M": SPREAD_HYPERBOLA,
    }
    assert spread_forms(assignment, "F") == {
        "V": SPREAD_HYPERBOLA,
        "P": SPREAD_LINE,
        "M": SPREAD_HYPERBOLA,
    }


def test_a_form_without_an_approximation_of_the_spread_is_an_error() -> None:
    config = LawAssignmentConfig.model_validate(
        {
            "quantities": {
                "H": {
                    "blocks": {},
                    "spread": {"V": "linear", "P": "logarithmic", "M": "fractional_hyperbolic"},
                }
            }
        }
    )
    with pytest.raises(ConfigError) as caught:
        spread_forms(build_law_assignment(config), "H")
    assert caught.value.error_code == LAW_FORM_UNSUPPORTED
