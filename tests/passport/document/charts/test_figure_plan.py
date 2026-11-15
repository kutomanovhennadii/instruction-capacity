"""The plan of the figures: the shipped plan, the names of the figures, the refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ConfigError, ContractViolationError
from iec.passport.document.charts.figure_plan import FIGURE_PLAN_CONFIG_INVALID, load_figure_plan
from iec.passport.document.charts.figure_writer import figure_names, write_passport_figures
from iec.passport.document.charts.law_views import FIGURE_VIEW_UNKNOWN
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config

CONFIG = Path(__file__).resolve().parents[4] / "config"
SHIPPED = CONFIG / "passport_figures.yaml"
ASSIGNMENT = build_law_assignment(load_law_assignment_config(CONFIG / "passport_laws.yaml"))
MEASURED = {"status": "measured", "fit": {}}
WITH_SPREAD = {"status": "measured", "fit": {"spread": {"V": [], "P": [], "M": []}}}


def test_the_shipped_plan_names_the_figures_of_the_passport() -> None:
    plan = load_figure_plan(SHIPPED)
    assert figure_names(plan, "P_sync", MEASURED) == [
        "S-before-correction",
        "S-after-correction",
        "plateau-before-correction",
        "P-before-correction",
        "P-after-correction",
        "M-before-correction",
        "M-after-correction",
    ]
    assert figure_names(plan, "H", MEASURED)[-2:] == ["R-before-correction", "R-after-correction"]
    assert figure_names(plan, "F", WITH_SPREAD) == [
        "V-before-correction",
        "V-after-correction",
        "answer-share-before-correction",
        "answer-share-after-correction",
        "P-before-correction",
        "P-after-correction",
        "M-before-correction",
        "M-after-correction",
        "spread-V",
        "spread-P",
        "spread-M",
    ]


def test_a_quantity_without_laws_has_no_figures() -> None:
    assert figure_names(load_figure_plan(SHIPPED), "H", {"status": "no_laws"}) == []


def _plan(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "passport_figures.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "text",
    [
        "quantities:\n  P_sync:\n    - {kind: axis, stem: P}\n",
        "quantities:\n  P_sync:\n    - {kind: pie, block: density, stem: P}\n",
        "quantities:\n  Q:\n    - {kind: spread, stem: spread}\n",
        "format_version: 2\nquantities: {}\n",
    ],
    ids=["axis-without-block", "unknown-kind", "unknown-quantity", "newer-format"],
)
def test_a_broken_plan_is_refused(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError) as caught:
        load_figure_plan(_plan(tmp_path, text))
    assert caught.value.error_code == FIGURE_PLAN_CONFIG_INVALID


def test_a_block_without_a_view_is_refused(tmp_path: Path) -> None:
    plan = load_figure_plan(
        _plan(tmp_path, "quantities:\n  F:\n    - {kind: axis, block: grid_slices, stem: S}\n")
    )
    passport = {"model": {"id": "model-x"}, "F": MEASURED}
    with pytest.raises(ContractViolationError) as caught:
        write_passport_figures(passport, tmp_path / "figures", ASSIGNMENT, plan)
    assert caught.value.error_code == FIGURE_VIEW_UNKNOWN
