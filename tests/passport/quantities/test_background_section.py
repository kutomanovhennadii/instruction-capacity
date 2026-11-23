"""The section of F: the trials of a point, the matrix, the laws and three capacities on known laws."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iec.answer_validator import align_answer, build_reference
from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.fitting.chain.fit_paths import METHOD_GLMM
from iec.passport.fitting.load_spread import LoadSpread, SpreadAxis
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.block_choice import BlockChoice, BlockLaw
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.laws.forms.law_fit import LawFit
from iec.passport.laws.forms.linear_forms import LogLinearLoadForm
from iec.passport.quantities.background_design import (
    BackgroundCells,
    answer_share_design,
    background_cells,
)
from iec.passport.quantities.background_section import (
    P_ZERO,
    _density_top,
    background_laws,
)
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin
from iec.point_execution import task_from_record

#: The forms of the laws of the new folder and the form of the law of the density with a hill.
ASSIGNMENT = build_law_assignment(
    load_law_assignment_config(Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml")
)
HILL = LogLinearLoadForm(P_ZERO)

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 5120),
    density_bounds=doubling_bounds(1, 256),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 256)),
)
SIGMA2 = 2.0
AT_20, KAPPA_V = 9.0, -0.4
Q_A, Q_TAU = 1.0, 0.5
P_A, P_B, P_C = 0.5, 0.2, -3.0
KAPPA_M = -0.3
P_TOP = P_B / (-P_C * math.log(2.0))


def _share(q: float) -> float:
    return Q_A * math.exp(-q / Q_TAU)


#: The law in the coding of the fit: the last tenth, its centre at 0.95, is the reference bin.
Q_AT_0 = _share(0.0) - _share(0.95)
Q_AT_1 = _share(1.0) - _share(0.95)


def _task(answer: str):
    source = "lake tree cat dog\nlate tree cut dog"
    record = PointRecord(
        conditions=PointConditions(D=6, n=1, m=2, seed_rules=1, seed_text=2, seed_order=0),
        rules=(("lake", "stone"), ("tree", "fish")),
        source_text=source,
        answer=answer,
        status=PointStatus.OK,
        prompt_tokens=None,
        response_tokens=None,
    )
    table, text = task_from_record(record, "v1_intersection")
    reference = build_reference(table, text)
    return table, text, align_answer(answer, reference, text, table)


def _points() -> list[BackgroundCells]:
    """Points over a grid of lengths, densities and rules; every bin has many requests with failures."""
    points = []
    for D in (8, 16, 32, 64, 128, 256):  # noqa: N806
        for n in (5, 10, 20):
            for m in (1, 2, 4, 8):
                if m > D:
                    continue
                for repeat in range(11):
                    cells = tuple((share, 12, 6) for share in range(1, 11))
                    points.append(
                        BackgroundCells(key=f"{D}/{n}/{m}/{repeat}", D=D, n=n, m=m, cells=cells)
                    )
    return points


def _density(p: float, b: float = P_B) -> float:
    return P_A + b * math.log2(p / P_ZERO) + P_C * p


def _known_fit(at_20: float = AT_20, share_sign: float = 1.0, density_b: float = P_B):
    built = answer_share_design(_points(), LAYOUT, {point.key: 1.0 for point in _points()})
    centers = built.centers
    reference_p = centers["P"][built.references["P"]]
    theta = []
    for name in built.design.names:
        if name.startswith("v_bin_"):
            value = at_20 + KAPPA_V * math.log2(centers["V"][name] / 20.0)
        elif name.startswith("ql_bin_"):
            value = share_sign * (_share(centers["q"][name]) - _share(0.95))
        elif name.startswith("p_bin_"):
            value = _density(centers["P"][name], density_b) - _density(reference_p, density_b)
        else:
            value = KAPPA_M * math.log2(centers["M"][name])
        theta.append(value)
    size = len(theta)
    covariance = np.eye(size) * 1e-4
    fit = VariableFit(
        variable="F",
        method=METHOD_GLMM,
        names=built.design.names,
        theta=tuple(theta),
        se=tuple(float(value) for value in np.sqrt(np.diag(covariance))),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        dropped=(),
        sigma2=SIGMA2,
        sigma2_covariance=(*([0.0] * size), 0.01),
        steps=1,
        rows=1,
        points=1,
        failures=10,
        trials=100,
    )
    top = P_TOP if density_b > 0.0 else P_ZERO
    return built, fit, _density(top, density_b) - _density(reference_p, density_b)


def test_cells_count_background_words_on_their_places() -> None:
    table, text, aligned = _task("stone fish cat dog\nlate fish stone dog")
    point = background_cells("k", 6, 1, 2, aligned, text, table)
    assert sum(cell[1] for cell in point.cells) == 5
    # The misfire: "stone" on the place of "cut".
    assert sum(cell[2] for cell in point.cells) == 1


def test_share_of_the_answer_goes_by_equal_bins() -> None:
    table, text, aligned = _task("stone fish cat dog\nlate fish cut dog")
    point = background_cells("k", 6, 1, 2, aligned, text, table)
    # Background words stand at 3, 4, 5, 7, 8 of 8 words: shares 3/8 … 1.
    assert sorted({cell[0] for cell in point.cells}) == [4, 6, 7, 9, 10]


def test_matrix_folds_rows_and_leaves_out_reference_bins() -> None:
    points = _points()
    built = answer_share_design(points, LAYOUT, {point.key: 1.0 for point in points})
    names = built.design.names
    assert "ql_bin_10" not in names
    assert "m_bin_1" not in names
    assert built.references["P"] not in names
    assert built.design.x.sum(axis=1).max() <= 4.0
    assert set(built.bins) == {"V", "q", "P", "M"}
    assert built.bins["M"]["m_bin_1"].effective == pytest.approx(11.0 * 18.0)


def test_point_without_a_weight_is_an_error() -> None:
    with pytest.raises(ContractViolationError) as caught:
        answer_share_design(_points(), LAYOUT, {})
    assert caught.value.error_code == "POINT_WEIGHT_MISSING"


def test_point_without_trials_gives_no_rows() -> None:
    points = [BackgroundCells("a", 8, 5, 8, ())]
    built = answer_share_design(points, LAYOUT, {"a": 1.0})
    assert built.design.point_keys == ()


def test_laws_recover_known_lines() -> None:
    built, fit, p_zero = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    assert section["status"] == "measured"
    assert section["model"] == "V-(s/v)-PM"
    laws = section["laws"]
    assert set(laws) == {"V", "q", "P", "M"}
    # The data choose the forms they were drawn from.
    assert [laws[letter]["law_form"] for letter in "VqPM"] == [
        "linear",
        "exponential",
        "log_linear_load",
        "linear",
    ]
    assert laws["V"]["at_20"] == pytest.approx(AT_20)
    assert laws["V"]["params"]["b"] == pytest.approx(KAPPA_V)
    assert laws["q"]["params"]["A"] == pytest.approx(Q_A, abs=1e-6)
    assert laws["q"]["params"]["tau"] == pytest.approx(Q_TAU, abs=1e-6)
    assert laws["q"]["at_0"] == pytest.approx(Q_AT_0, abs=1e-6)
    assert laws["q"]["at_1"] == pytest.approx(Q_AT_1, abs=1e-6)
    assert laws["q"]["held"] == []
    assert laws["M"]["params"]["b"] == pytest.approx(KAPPA_M)
    assert laws["P"]["params"]["b"] == pytest.approx(P_B)
    assert laws["P"]["params"]["c"] == pytest.approx(P_C)
    alpha = AT_20 + Q_AT_0 + p_zero
    assert section["alpha"]["value"] == pytest.approx(alpha, abs=1e-6)
    assert section["alpha"]["half"] > 0.0


def test_three_capacities_follow_the_formulas() -> None:
    built, fit, p_zero = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    alpha = AT_20 + Q_AT_0 + p_zero
    # At the level 0.5 the threshold is zero at any variance, so the reserve is the level itself.
    reserve = alpha
    assert "capacity_level" not in section
    rows = {(row["key"], row["method"]): row for row in section["capacities"]}
    assert set(rows) == {
        ("instruction_capacity", "law"),
        ("instruction_capacity", "bins"),
        ("fatigue_capacity_words", "law"),
    }
    assert all(row["level"] == 0.5 for row in section["capacities"])
    assert rows[("instruction_capacity", "law")]["value"] == pytest.approx(
        2.0 ** (reserve / -KAPPA_M)
    )
    fatigue = 20.0 * 2.0 ** ((reserve - Q_AT_0 + Q_AT_1) / -KAPPA_V)
    assert rows[("fatigue_capacity_words", "law")]["value"] == pytest.approx(fatigue, rel=1e-5)


def test_the_law_covariance_follows_the_published_laws() -> None:
    built, fit, _ = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    laws = section["laws"]
    published = {letter: laws[letter] for letter in "VqPM"}
    entry = section["law_covariance"]
    assert [part["part"] for part in entry["parts"]] == ["V", "q", "P", "M"]
    names = [(part["part"], name) for part in entry["parts"] for name in part["names"]]
    assert entry["vector"] == pytest.approx(
        [published[key]["params"][name] for key, name in names]
    )
    matrix = np.asarray(entry["matrix"])
    assert np.allclose(matrix, matrix.T)
    # The diagonal of R·J·Rᵀ gives the published half-widths: 1.96 standard errors of every parameter.
    halves = [published[key]["half"][name] for key, name in names]
    assert 1.96 * np.sqrt(np.diag(matrix)) == pytest.approx(halves, rel=1e-6, abs=1e-12)


def test_capacity_by_the_bins_interpolates_inside_the_bins_of_m() -> None:
    built, fit, _ = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    rows = {(row["key"], row["method"]): row for row in section["capacities"]}
    assert rows[("instruction_capacity", "bins")]["status"] in {"measured", "beyond_support"}
    assert section["edge_rules"] == 8
    assert section["edge_words"] == 5120


def test_rows_keep_the_reference_bin_with_an_own_error() -> None:
    built, fit, _ = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    rows = {row["name"]: row for row in section["bins"]["M"]}
    assert rows["m_bin_1"]["status"] == "reference"
    assert rows["m_bin_1"]["raw"] == 0.0
    assert rows["m_bin_1"]["own_half"] > 0.0
    assert set(section["bins"]) == {"V", "q", "P", "M"}


def test_a_rising_share_law_is_the_exponential_of_a_negative_amplitude() -> None:
    # The exponential takes A of either sign: the rising law of the data is its own form.
    built, fit, _ = _known_fit(share_sign=-1.0)
    section = background_laws(built, fit, ASSIGNMENT)
    assert section["status"] == "measured"
    share = section["laws"]["q"]
    assert share["law_form"] == "exponential"
    assert share["params"]["A"] < 0.0
    assert share["at_1"] > share["at_0"]


@pytest.mark.parametrize(
    ("b", "c", "top"),
    [
        (0.2, -3.0, 0.2 / (3.0 * math.log(2.0))),
        (-0.1, -3.0, P_ZERO),
        (0.0, -3.0, P_ZERO),
        (0.2, 0.5, 1.0),
        (5.0, -1.0, 1.0),
    ],
)
def test_the_top_of_the_density_law(b: float, c: float, top: float) -> None:
    params = np.array([0.0, b, c])
    fit = LawFit(HILL.name, params, 0.0, (), np.zeros((3, 0)), True)
    law = BlockLaw(BlockChoice(HILL, fit, ()), np.zeros((3, 0)), (), np.array([0.0, 10.0]), False)
    assert P_ZERO * 2.0 ** _density_top(law) == pytest.approx(top, rel=1e-6)


def test_the_shift_of_the_density_is_the_top_of_its_hill() -> None:
    built, fit, top_value = _known_fit()
    section = background_laws(built, fit, ASSIGNMENT)
    density = section["laws"]["P"]
    assert density["p_top"] == pytest.approx(P_TOP)
    assert density["at_top"] == pytest.approx(top_value)
    assert section["shifts"]["delta_P"]["value"] == pytest.approx(top_value)
    # After the correction every contribution of the load is a price: the law stays at or below zero.
    assert all(row["law"] <= density["at_top"] + 1e-9 for row in section["bins"]["P"] if "law" in row)


def test_a_law_without_a_hill_keeps_the_shift_at_the_smallest_density() -> None:
    built, fit, top_value = _known_fit(density_b=-0.1)
    density = background_laws(built, fit, ASSIGNMENT)["laws"]["P"]
    assert density["p_top"] == P_ZERO
    assert density["at_top"] == pytest.approx(top_value)


def _with_rules_spread(fit: VariableFit) -> VariableFit:
    """The spread of two groups of the number of rules: σ² = 2 at one rule, smaller at fifty."""
    single = SpreadAxis(((1,),), (None,), (1.0,), (1.0,), (40,))
    rules = SpreadAxis(((1,), (2,)), (None, 1), (1.0, 50.0), (0.5, 0.5), (40, 40))
    size = len(fit.names)
    return replace(
        fit,
        method="glmm_heterogeneous",
        spread=LoadSpread((0.5 * math.log(2.0), -0.6), (single, single, rules)),
        spread_covariance=tuple(tuple(row) for row in (np.eye(size + 2) * 1e-4).tolist()),
    )


def test_the_capacities_of_f_do_not_depend_on_the_spread() -> None:
    built, fit, _ = _known_fit()
    plain = background_laws(built, fit, ASSIGNMENT)
    section = background_laws(built, _with_rules_spread(fit), ASSIGNMENT)
    assert section["capacities"] == plain["capacities"]
    assert all("sigma2" not in row and "threshold" not in row for row in section["capacities"])
    assert "capacity_level" not in section
