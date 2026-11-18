"""The section of P_sync: its matrix, the laws and capacities on known laws."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iec.passport.fitting.chain.fit_paths import METHOD_GLMM, METHOD_NONE
from iec.passport.fitting.glmm_fit import marginal_threshold
from iec.passport.fitting.load_spread import LoadSpread, SpreadAxis
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.forms.hyperbolic_forms import AsymptoticHyperbolicForm, SlantedHyperbolicForm
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.laws.plateau_grid import vs_curve
from iec.passport.quantities.sync_section import (
    P_ZERO,
    _log_word_root,
    sync_design,
    sync_laws,
)
from iec.passport.trials.design.folded_design import AxisBin, AxisTable, GridCell, VariableDesign
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin
from iec.passport.trials.place_trials import PointTrials

#: The forms of the laws of the new folder and the form of the plateau and of the density.
ASSIGNMENT = build_law_assignment(
    load_law_assignment_config(Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml")
)
HYPERBOLA = SlantedHyperbolicForm()

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 10240),
    density_bounds=doubling_bounds(1, 512),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 512)),
)
#: κ_S, λ_S, ω_S of the slices and μ₀, κ_V, u*, ω_V of the plateau of the known grid law.
SLICES = np.array([-3.0, 40.0, 1.5])
PLATEAU = np.array([20.0, -3.0, 5.3, 0.001])
#: μ_P, κ_P, u*, ω_P of the known density law.
DENSITY = np.array([10.0, -5.0, 8.0, 2.0])
#: λ_M, κ_M of the known law of the number of rules.
RULES = np.array([0.5, -1.0])
SIGMA2 = 4.0


def _point(key: str, D: int, n: int, m: int) -> PointTrials:  # noqa: N803
    return PointTrials(
        key=key,
        D=D,
        n=n,
        m=m,
        sync=((1, 10, 1, 21.8),),
        rule=(),
        background=(),
        totals={},
    )


def _density(u: np.ndarray) -> np.ndarray:
    distance = u - DENSITY[2]
    return DENSITY[0] + 0.5 * DENSITY[1] * (
        distance + np.sqrt(distance * distance + DENSITY[3] ** 2)
    )


def _known_fit() -> tuple[VariableDesign, VariableFit]:
    """Coefficients exactly on the known laws, with small independent errors."""
    cells: list[GridCell] = []
    abscissae: dict[str, dict[str, float]] = {}
    theta: list[float] = []
    names: list[str] = []
    for v_bin in range(3, 11):
        lower, upper = LAYOUT.volume_bounds[v_bin - 1]
        u = v_bin - 2.0
        for s_bin in range(1, v_bin + 2):
            name = f"vs_{v_bin}_{s_bin}"
            x = 5.0 * 2.0 ** (s_bin - 1)
            cells.append(GridCell(v_bin, s_bin, name, lower, upper, 100, 10, 20))
            abscissae[name] = {"x": x, "u": u}
            names.append(name)
            theta.append(
                float(
                    vs_curve(
                        AsymptoticHyperbolicForm(), SLICES, HYPERBOLA, PLATEAU, np.array([[math.log2(x), u]])
                    )[0]
                )
            )
    reference = 8
    bins = []
    for number in range(1, 11):
        name = f"p_bin_{number}"
        p = 2.0 ** (number - 11 + 0.5) if number < 10 else 1.0
        abscissae[name] = {"p": p}
        requests = 3 if number == 1 else 20
        bins.append(AxisBin(number, p / 1.5, p * 1.5, name, 100, 10, requests))
        if number != reference:
            names.append(name)
            theta.append(float(_density(np.array([math.log2(p / P_ZERO)]))[0]))
    m_bins = []
    for number in range(1, 6):
        name = f"m_bin_{number}"
        m = 2.0 ** (number - 1)
        abscissae[name] = {"m": m}
        m_bins.append(AxisBin(number, m, m, name, 100, 10, 20))
        if number != 1:
            names.append(name)
            theta.append(float(RULES[0] + RULES[1] * math.log2(m)))
    size = len(names)
    design = VariableDesign(
        variable="P_sync",
        x=np.zeros((1, size)),
        failures=np.zeros(1),
        trials=np.ones(1),
        run=np.zeros(1, dtype=np.int64),
        point_keys=("a",),
        names=tuple(names),
        cells=tuple(cells),
        axes={"P": AxisTable("P", tuple(bins), reference), "M": AxisTable("M", tuple(m_bins), 1)},
        abscissae=abscissae,
    )
    covariance = np.eye(size) * 1e-4
    fit = VariableFit(
        variable="P_sync",
        method=METHOD_GLMM,
        names=tuple(names),
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
    return design, fit


def test_matrix_keeps_the_columns_of_the_number_of_rules() -> None:
    points = [_point("a", 40, 1, 2), _point("b", 40, 4, 1)]
    design = sync_design(points, LAYOUT, {"a": 1.0, "b": 1.0})
    assert [name for name in design.names if name.startswith("m_bin_")]
    assert set(design.axes) == {"P", "M"}
    assert [name for name in design.names if name.startswith("vs_")]


def test_the_laws_take_the_forms_of_their_data() -> None:
    design, fit = _known_fit()
    laws = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)["laws"]
    density = laws["density"]
    assert density["law_form"] == "slanted_hyperbolic"
    assert [density["params"][name] for name in HYPERBOLA.parameters] == pytest.approx(DENSITY, abs=1e-3)
    assert density["chi2"] == pytest.approx(0.0, abs=1e-6)
    assert laws["grid_plateau"]["law_form"] == "slanted_hyperbolic"
    assert laws["table_size"]["law_form"] == "linear"
    assert density["established_form"] == "slanted_hyperbolic"
    # The simple hyperbola is no candidate: its pole lies at the zero of the abscissa.
    assert {entry["form"] for entry in density["candidates"]} == {
        "slanted_hyperbolic",
        "linear",
        "binary_exponential",
    }


def test_laws_level_and_capacities_on_known_laws() -> None:
    design, fit = _known_fit()
    section = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)
    assert section["status"] == "measured"
    grid = section["laws"]["grid"]
    plateau = section["laws"]["grid_plateau"]["params"]
    assert grid["law_form"] == "asymptotic_hyperbolic"
    assert grid["params"]["kappa"] == pytest.approx(-3.0, abs=0.02)
    assert plateau["mu"] == pytest.approx(20.0, abs=0.05)
    assert plateau["kappa"] == pytest.approx(-3.0, abs=0.02)
    assert section["shifts"]["delta_VS"]["value"] == pytest.approx(20.0, abs=0.05)
    assert section["shifts"]["delta_P"]["value"] == pytest.approx(10.0, abs=0.05)
    assert section["alpha"]["value"] == pytest.approx(30.5, abs=0.1)
    assert section["alpha"]["half"] > 0.0
    by_key = {(row["key"], row["level"]): row for row in section["capacities"]}
    assert set(by_key) == {
        ("instruction_capacity", 0.5),
        ("fatigue_capacity_words", 0.5),
        ("density_odds_factor", None),
    }
    # At the level 0.5 the threshold of the logit of the median request is zero at any variance.
    threshold = 0.0
    # The fatigue capacity: the full law of the grid at p → 0 and v = s reaches the threshold.
    slices = np.array([grid["params"][name] for name in ("kappa", "lambda", "omega")])
    plateau_params = np.array([plateau[name] for name in HYPERBOLA.parameters])
    mu_p = section["shifts"]["delta_P"]["value"]
    lambda_m = section["shifts"]["delta_M"]["value"]
    fatigue = by_key[("fatigue_capacity_words", 0.5)]["value"]
    log_s = math.log2(fatigue)
    point = np.array([[log_s, max(log_s - math.log2(20.0), 0.0)]])
    value = vs_curve(AsymptoticHyperbolicForm(), slices, HYPERBOLA, plateau_params, point)[0]
    assert value + mu_p + lambda_m == pytest.approx(threshold, abs=1e-6)
    assert by_key[("fatigue_capacity_words", 0.5)]["interval"][0] < fatigue
    # Full density multiplies the odds of an error by e^(−(P(1) − δ_P)): the factor keeps the sign.
    odds = by_key[("density_odds_factor", None)]
    change = section["laws"]["density"]["at_one"] - mu_p
    assert odds["value"] == pytest.approx(math.exp(-change))
    assert odds["change"] == pytest.approx(change)
    assert odds["interval"] == pytest.approx(
        [math.exp(-change - odds["change_half"]), math.exp(-change + odds["change_half"])]
    )


def test_rows_keep_the_reference_bin_with_an_own_error_and_every_bin_on_the_law() -> None:
    design, fit = _known_fit()
    section = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)
    rows = {row["name"]: row for row in section["bins"]["P"]}
    assert rows["p_bin_8"]["status"] == "reference"
    assert rows["p_bin_8"]["raw"] == 0.0
    assert rows["p_bin_8"]["own_half"] > 0.0
    # A bin of few requests lies on the law and is not an outlier: it enters the law.
    assert rows["p_bin_1"]["status"] == "law"
    assert rows["p_bin_5"]["status"] == "law"
    assert rows["p_bin_5"]["restored"] == pytest.approx(
        rows["p_bin_5"]["raw"] - section["shifts"]["delta_P"]["value"]
    )


def test_a_fit_that_gave_nothing_gives_no_laws() -> None:
    design, fit = _known_fit()
    empty = VariableFit(
        variable="P_sync",
        method=METHOD_NONE,
        names=(),
        theta=(),
        se=(),
        covariance=(),
        dropped=(),
        sigma2=None,
        sigma2_covariance=(),
        steps=0,
        rows=0,
        points=0,
        failures=0,
        trials=0,
    )
    section = sync_laws(design, empty, LAYOUT, 10240.0, ASSIGNMENT)
    assert section["status"] == "no_laws"
    assert "capacities" not in section


def test_fatigue_capacity_is_one_word_when_the_first_word_is_below_the_threshold() -> None:
    assert _log_word_root(lambda x: -100.0 - x, marginal_threshold(0.95, SIGMA2)[0]) == 0.0


def test_a_law_that_rises_with_the_length_has_no_fatigue_capacity() -> None:
    # The plateau rises with the length faster than the slices fall: the law never reaches zero.
    assert math.isnan(_log_word_root(lambda x: 1.0 + 0.3 * x, 0.0))
    assert _log_word_root(lambda x: 5.0 - x, 0.0) == pytest.approx(5.0)


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


def test_the_capacities_of_p_sync_do_not_depend_on_the_spread() -> None:
    design, fit = _known_fit()
    plain = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)
    spread = sync_laws(design, _with_rules_spread(fit), LAYOUT, 10240.0, ASSIGNMENT)
    assert spread["capacities"] == plain["capacities"]
    assert {row["level"] for row in spread["capacities"] if row["key"] != "density_odds_factor"} == {0.5}
    assert all("sigma2" not in row and "threshold" not in row for row in spread["capacities"])
    assert "thresholds" not in spread


def test_the_law_of_the_number_of_rules_enters_the_level_and_the_rows() -> None:
    design, fit = _known_fit()
    section = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)
    table = section["laws"]["table_size"]
    assert section["model"] == "VS-PM"
    assert table["params"]["a"] == pytest.approx(RULES[0], abs=1e-6)
    assert table["params"]["b"] == pytest.approx(RULES[1], abs=1e-6)
    assert table["n_points"] == 4
    lambda_m = section["shifts"]["delta_M"]["value"]
    assert lambda_m == pytest.approx(table["params"]["a"])
    rows = {row["name"]: row for row in section["bins"]["M"]}
    assert rows["m_bin_1"]["status"] == "reference"
    assert rows["m_bin_3"]["status"] == "law"
    assert rows["m_bin_3"]["restored"] == pytest.approx(rows["m_bin_3"]["raw"] - lambda_m)
    assert rows["m_bin_3"]["law"] == pytest.approx(RULES[0] + RULES[1] * 2.0, abs=1e-6)


def test_the_instruction_capacity_of_p_sync_follows_the_law_of_the_number_of_rules() -> None:
    design, fit = _known_fit()
    section = sync_laws(design, fit, LAYOUT, 10240.0, ASSIGNMENT)
    rows = {
        row["method"]: row for row in section["capacities"] if row["key"] == "instruction_capacity"
    }
    kappa_m = section["laws"]["table_size"]["params"]["b"]
    by_law = rows["law"]
    assert math.log2(by_law["value"]) == pytest.approx(-section["alpha"]["value"] / kappa_m)
    assert by_law["interval"][0] < by_law["value"] < by_law["interval"][1]
    assert by_law["status"] == "extrapolated"
    # The bins fall by one logit per doubling up to sixteen rules: the reserve of 30.5 is not spent.
    assert rows["bins"]["status"] == "beyond_support"
    assert rows["bins"]["lower_bound"] == 16
    assert rows["bins"]["reserve_left"] == pytest.approx(30.5 - 4.0, abs=0.1)


def test_a_flat_law_of_the_number_of_rules_gives_no_instruction_capacity() -> None:
    design, fit = _known_fit()
    theta = tuple(
        float(RULES[0]) if name.startswith("m_bin_") else value
        for name, value in zip(fit.names, fit.theta, strict=True)
    )
    section = sync_laws(design, replace(fit, theta=theta), LAYOUT, 10240.0, ASSIGNMENT)
    by_law = next(
        row
        for row in section["capacities"]
        if row["key"] == "instruction_capacity" and row["method"] == "law"
    )
    assert by_law["value"] is None
    assert by_law["status"] == "no_slope"
