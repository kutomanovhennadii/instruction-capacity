"""The standard stress test: the point, the structure, the influences, the intervals and the statuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.laws.grid_slices import SLICE_LEVEL
from iec.passport.stress_test import stress_test_section
from iec.passport.stress_test.stress_test import NAMES, PARTS

ASSIGNMENT = build_law_assignment(
    load_law_assignment_config(
        Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml"
    )
)


def _line(a: float, b: float) -> dict[str, Any]:
    """Return a published line ``a + b·x``."""
    return {"law_form": "linear", "params": {"a": a, "b": b}}


def _flat_slices() -> dict[str, Any]:
    """Return slices that keep their level along the answer: a binary exponential without an amplitude."""
    return {"law_form": "binary_exponential", "params": {"A": 0.0, "gamma": 0.5}}


def _section(quantity: str, laws: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Return a section with laws and its law covariance from the published parameters, the covariance 0.01·I."""
    parts: list[dict[str, Any]] = []
    vector: list[float] = []
    for key, law_key, block in PARTS[quantity]:
        law = laws[law_key]
        form = next(
            item for item in ASSIGNMENT.forms_of(quantity, block) if item.name == law["law_form"]
        )
        names = [name for name in form.parameters if not (key == "slices" and name == SLICE_LEVEL)]
        parts.append({"part": key, "names": names})
        vector += [float(law["params"][name]) for name in names]
    matrix = (0.01 * np.eye(len(vector))).tolist()
    return {
        "status": "measured",
        "laws": laws,
        "law_covariance": {"parts": parts, "vector": vector, "matrix": matrix},
        **extra,
    }


def _passport(h_level: float = 10.0, rule_bin: int = 9, sync_rules: float = -0.1) -> dict[str, Any]:
    """Return a passport whose H falls by one logit per doubling of the text and per doubling of the rules.

    P_sync stays near one, so the combined error is the error of H: the logit of the last word is
    ``h_level − log₂(2.5·m) − log₂ m``, zero at ``log₂ m = (h_level − log₂ 2.5)/2``.
    """
    sync = _section(
        "P_sync",
        {
            "grid": _flat_slices(),
            "grid_plateau": _line(50.0, -0.1),
            "density": _line(0.0, 0.0),
            "table_size": _line(0.0, sync_rules),
        },
    )
    rule = _section(
        "H",
        {
            "grid": _flat_slices(),
            "grid_plateau": _line(h_level, -1.0),
            "density": _line(0.0, 0.0),
            "table_size": _line(0.0, -1.0),
            "rule_position": _line(0.0, 0.0),
        },
        bins={
            "grid": [{"words": [5121, 10240], "requests": 3}],
            "M": [{"bin": rule_bin, "requests": 2}],
        },
    )
    background = _section(
        "F",
        {
            "V": _line(12.0, -1.0),
            "q": _line(0.0, 0.0),
            "P": {**_line(0.0, 0.0), "p_top": 2.0**-10},
            "M": _line(0.0, -1.0),
        },
    )
    return {"P_sync": sync, "H": rule, "F": background}


def test_the_point_is_where_the_last_word_fails_in_half_of_the_cases() -> None:
    section = stress_test_section(_passport(), ASSIGNMENT)
    assert section["status"] == "measured"
    assert section["support"] == {"words": 10240.0, "rules": 256.0}
    assert section["rules"]["value"] == pytest.approx(20.2386, abs=0.01)
    assert section["words"]["value"] == pytest.approx(1011.9, abs=0.5)
    assert section["combined_error_at_point"] == 50.0


def test_the_structure_of_the_errors_and_the_errors_over_the_text() -> None:
    section = stress_test_section(_passport(), ASSIGNMENT)
    structure = section["structure"]
    # P_sync stays at one: every combined error is a replacement.
    assert structure["synchronization"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert structure["replacements"]["value"] == pytest.approx(100.0, abs=1e-6)
    # The logit of F at the point is 12 − 10 = 2: 1 − σ(2) = 0.1192 of the background words per 0.5 combined errors.
    assert structure["background"]["value"] == pytest.approx(23.84, abs=0.1)
    # The slices are flat: every word of the text fails as the last one.
    assert section["replacements"]["text_errors"]["value"] == pytest.approx(50.0, abs=0.05)
    assert section["combined_text_errors"]["value"] == pytest.approx(50.0, abs=0.05)
    assert section["background"]["text_errors"]["value"] == pytest.approx(11.92, abs=0.05)


def test_the_influence_is_the_share_of_the_price_of_a_load() -> None:
    section = stress_test_section(_passport(), ASSIGNMENT)
    for kind in ("synchronization", "replacements", "background"):
        influence = section[kind]["influence"]
        assert influence["length"]["value"] == pytest.approx(56.61, abs=0.05)
        assert influence["rules"]["value"] == pytest.approx(43.39, abs=0.05)
        assert influence["density"]["value"] == pytest.approx(0.0, abs=1e-9)


def test_a_load_that_lowers_the_errors_has_a_negative_share() -> None:
    # Synchronization grows with the rules: the price of rules is −0.05·log₂ m against 0.1·u of length.
    influence = stress_test_section(_passport(sync_rules=0.05), ASSIGNMENT)["synchronization"][
        "influence"
    ]
    assert influence["rules"]["value"] == pytest.approx(-62.1, abs=0.2)
    assert influence["length"]["value"] == pytest.approx(162.1, abs=0.2)
    shares = [influence[load]["value"] for load in ("length", "density", "rules")]
    assert sum(shares) == pytest.approx(100.0, abs=1e-6)


def test_the_influence_is_not_defined_when_the_loads_lower_the_errors() -> None:
    # The price of rules −log₂ m outweighs 0.1·u of length: synchronization is better at the point than at the zero.
    influence = stress_test_section(_passport(sync_rules=1.0), ASSIGNMENT)["synchronization"][
        "influence"
    ]
    assert all(influence[load]["value"] is None for load in ("length", "density", "rules"))


def test_every_number_has_a_finite_half_width() -> None:
    section = stress_test_section(_passport(), ASSIGNMENT)
    rows = [
        section["words"],
        section["rules"],
        section["combined_text_errors"],
        *section["structure"].values(),
    ]
    for kind in ("synchronization", "replacements", "background"):
        rows += [section[kind]["text_errors"], *section[kind]["influence"].values()]
    assert len(rows) == len(NAMES)
    assert all(row["half"] is not None and row["half"] >= 0.0 for row in rows)
    assert section["rules"]["half"] > 0.0


def test_the_test_reads_the_same_from_a_file_with_sorted_keys() -> None:
    # The file of the passport sorts the keys of every dictionary: the parts keep their order in a list.
    passport = _passport()
    stored = json.loads(json.dumps(passport, sort_keys=True))
    assert stress_test_section(stored, ASSIGNMENT) == stress_test_section(passport, ASSIGNMENT)


def test_a_point_past_the_measured_rules_is_a_continuation_of_the_laws() -> None:
    assert stress_test_section(_passport(rule_bin=4), ASSIGNMENT)["status"] == "extrapolated"


def test_the_statuses_without_a_point() -> None:
    assert stress_test_section(_passport(h_level=1.0), ASSIGNMENT)["status"] == "below_one_rule"
    assert stress_test_section(_passport(h_level=1000.0), ASSIGNMENT)["status"] == "beyond_range"
    passport = _passport()
    passport["F"] = {"status": "no_laws"}
    assert stress_test_section(passport, ASSIGNMENT) == {
        "protocol": {"density": 0.5, "words_per_rule": 50, "point_error": 0.5},
        "status": "no_laws",
    }
