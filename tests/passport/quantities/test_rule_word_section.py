"""The section of H: the cumulative price, the fatigue root, the bins, the laws on known laws."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iec.passport.fitting.load_spread import LoadSpread, SpreadAxis
from iec.passport.laws.block_choice import BlockChoice, BlockLaw
from iec.passport.laws.forms.growth_forms import BinaryExponentialForm, SaturationForm
from iec.passport.laws.forms.law_fit import LawFit
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.quantities.rule_word_section import (
    P_ZERO,
    capacity_by_bins,
    log_capacity,
    log_fatigue,
    rule_word_laws,
)
from iec.passport.trials.design.folded_design import AxisBin, AxisTable, GridCell, VariableDesign
from iec.passport.fitting.chain.fit_paths import METHOD_GLMM
from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.laws.forms.hyperbolic_forms import SlantedHyperbolicForm
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config

#: The forms of the laws of the new folder and the form of the law of the rule number.
ASSIGNMENT = build_law_assignment(
    load_law_assignment_config(Path(__file__).resolve().parents[3] / "config" / "passport_laws.yaml")
)
RULE_FORM = SlantedHyperbolicForm()
LINE = LinearForm()


def _law(form: LawForm, params: np.ndarray) -> BlockLaw:
    """Return a law of a block with given parameters and no rows: the capacities read its curve."""
    values = np.asarray(params, dtype=float)
    fit = LawFit(form.name, values, 0.0, (), np.zeros((len(values), 0)), True)
    rows = np.zeros((len(values), 0))
    return BlockLaw(BlockChoice(form, fit, ()), rows, (), np.array([0.0, 1.0]), False)


#: A flat law of the rule number: the price of a table is the line of M alone.
FLAT_R = np.array([0.0, 0.0, 1.0, 1.0])
A, GAMMA, MU_0, KAPPA_V = 0.5, 0.3, 13.0, -0.9
LAMBDA_M, KAPPA_M = 0.3, -0.2
P_LAW = np.array([2.0, -2.5, 7.5, 1.0])
R_LAW = np.array([-0.4, -0.6, 3.8, 1.0])
SIGMA2 = 2.0


def _hyperbola(params: np.ndarray, x: float) -> float:
    distance = x - params[2]
    return float(
        params[0] + 0.5 * params[1] * (distance + math.sqrt(distance * distance + params[3] ** 2))
    )


def _known_fit(common_error: float = 0.0) -> tuple[VariableDesign, VariableFit]:
    """Coefficients exactly on the known laws, with small independent errors."""
    names: list[str] = []
    theta: list[float] = []
    cells: list[GridCell] = []
    abscissae: dict[str, dict[str, float]] = {}
    for v_bin in range(3, 11):
        u = v_bin - 2.0
        for s_bin in range(1, v_bin + 2):
            name = f"vs_{v_bin}_{s_bin}"
            x = 5.0 * 2.0 ** (s_bin - 1)
            log_x = math.log2(x)
            cells.append(GridCell(v_bin, s_bin, name, 21, 40, 100, 10, 20))
            abscissae[name] = {"x": x, "u": u}
            names.append(name)
            theta.append(MU_0 + KAPPA_V * u - A * 2.0 ** (GAMMA * log_x))
    axes: dict[str, AxisTable] = {}
    p_bins = []
    for number in range(1, 11):
        name = f"p_bin_{number}"
        p = 2.0 ** (number - 11 + 0.5) if number < 10 else 1.0
        abscissae[name] = {"p": p}
        p_bins.append(AxisBin(number, p / 1.5, p * 1.5, name, 100, 10, 2 if number == 1 else 20))
        if number != 8:
            names.append(name)
            theta.append(_hyperbola(P_LAW, math.log2(p / P_ZERO)))
    axes["P"] = AxisTable("P", tuple(p_bins), 8)
    for letter, prefix, key in (("M", "m_bin_", "m"), ("R", "r_bin_", "r")):
        bins = []
        for number, (lower, upper) in enumerate(
            ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)), start=1
        ):
            name = f"{prefix}{number}"
            center = math.sqrt(lower * upper)
            abscissae[name] = {key: center}
            bins.append(AxisBin(number, float(lower), float(upper), name, 100, 10, 20))
            if number > 1:
                names.append(name)
                log_center = math.log2(center)
                theta.append(
                    LAMBDA_M + KAPPA_M * log_center
                    if letter == "M"
                    else _hyperbola(R_LAW, log_center)
                )
        axes[letter] = AxisTable(letter, tuple(bins), 1)
    size = len(names)
    design = VariableDesign(
        variable="H",
        x=np.zeros((1, size)),
        failures=np.zeros(1),
        trials=np.ones(1),
        run=np.zeros(1, dtype=np.int64),
        point_keys=("a",),
        names=tuple(names),
        cells=tuple(cells),
        axes=axes,
        abscissae=abscissae,
    )
    covariance = np.eye(size) * 1e-4
    cell_columns = [k for k, name in enumerate(names) if name.startswith("vs_")]
    covariance[np.ix_(cell_columns, cell_columns)] += common_error
    fit = VariableFit(
        variable="H",
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


def test_capacity_with_a_flat_rule_number_is_the_line_of_m() -> None:
    table, rule = _law(LINE, np.array([0.0, -1.0])), _law(RULE_FORM, FLAT_R)
    assert log_capacity(table, table.params, rule, rule.params, 3.0) == pytest.approx(3.0, abs=1e-3)


def test_capacity_counts_the_integral_over_the_rule_number() -> None:
    # R falls by one logit per doubling from the first rule: R(1) − R(r) = log₂ r.
    table = _law(LINE, np.array([0.0, 0.0]))
    rule = _law(RULE_FORM, np.array([0.0, -1.0, 0.0, 0.001]))
    x = log_capacity(table, table.params, rule, rule.params, 5.0)
    m = 2.0**x
    assert m * math.log2(m) - (m - 1.0) / math.log(2.0) == pytest.approx(5.0, abs=0.01)


def test_capacity_of_a_curved_law_of_the_number_of_rules_is_its_root() -> None:
    # A saturating M: the price of m rules is A·(1 − e^(−x/x_c)), x = log₂ m.
    table = _law(SaturationForm(), np.array([0.0, 4.0, 2.0]))
    rule = _law(RULE_FORM, FLAT_R)
    x = log_capacity(table, table.params, rule, rule.params, 2.0)
    assert 4.0 * (1.0 - math.exp(-x / 2.0)) == pytest.approx(2.0, abs=1e-3)


def test_capacity_without_a_reserve_is_below_one_rule() -> None:
    table, rule = _law(LINE, np.array([0.3, -0.5])), _law(RULE_FORM, FLAT_R)
    assert log_capacity(table, table.params, rule, rule.params, -1.0) == pytest.approx(-2.0)


def _fatigue(
    amplitude: float, gamma: float, kappa_v: float, alpha: float, threshold: float
) -> float | None:
    plateau = _law(LINE, np.array([0.0, kappa_v]))
    return log_fatigue(BinaryExponentialForm(), np.array([amplitude, gamma]), plateau, plateau.params, alpha, threshold)


def test_fatigue_below_twenty_words_is_the_root_of_the_exponent() -> None:
    # 2^(0.5·x) = 2 gives x = 2: four words, below twenty, so the length has no price.
    assert _fatigue(1.0, 0.5, -1.0, 5.0, 3.0) == pytest.approx(2.0)


def test_fatigue_past_twenty_words_counts_the_length() -> None:
    x = _fatigue(0.5, 0.3, -1.0, 30.0, 0.0)
    assert x is not None
    assert x > math.log2(20.0)
    assert 30.0 - 0.5 * 2.0 ** (0.3 * x) - (x - math.log2(20.0)) == pytest.approx(0.0, abs=1e-9)


def test_fatigue_of_a_law_that_does_not_fall_has_no_root() -> None:
    assert _fatigue(0.0, 0.5, 0.0, 10.0, 0.0) is None
    assert _fatigue(-1.0, 0.5, 0.0, 10.0, 0.0) is None


def test_fatigue_without_a_reserve_has_no_root() -> None:
    assert _fatigue(1.0, 0.5, -1.0, 3.0, 3.0) is None


def test_fatigue_past_the_end_of_the_search_has_no_root() -> None:
    assert _fatigue(1e-6, 0.005, 0.0, 30.0, 0.0) is None


def test_capacity_by_bins_walks_the_prices_of_rules() -> None:
    m_items = [
        AxisBin(2, 2.0, 2.0, "m_bin_2", 1, 1, 10),
        AxisBin(3, 3.0, 4.0, "m_bin_3", 1, 1, 10),
    ]
    theta = {"m_bin_2": -1.0, "m_bin_3": -3.0}
    # Rule 2 costs 1, rules 3 and 4 cost 1 each: the jump 2 is spread over two rules.
    assert capacity_by_bins(theta, m_items, [], 0.0, 0.0, 1.5)["value"] == pytest.approx(2.5)
    beyond = capacity_by_bins(theta, m_items, [], 0.0, 0.0, 5.0)
    assert beyond["status"] == "beyond_support"
    assert beyond["lower_bound"] == 4
    assert beyond["reserve_left"] == pytest.approx(2.0)
    assert capacity_by_bins(theta, m_items, [], 0.0, 0.0, -1.0)["status"] == "below_one_rule"


def test_laws_level_and_capacities_on_known_laws() -> None:
    design, fit = _known_fit()
    section = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)
    assert section["status"] == "measured"
    laws = section["laws"]
    assert laws["grid"]["params"]["A"] == pytest.approx(A, abs=1e-4)
    assert laws["grid"]["params"]["gamma"] == pytest.approx(GAMMA, abs=1e-4)
    assert laws["grid"]["held"] == []
    # The data choose the forms they were drawn from: the line, the line, the slanted hyperbola of
    # the rule number and of the density - the established two-slope hyperbola describes the density
    # as exactly with one parameter more, so the smallest QAIC takes the slanted hyperbola.
    assert laws["grid_plateau"]["law_form"] == "linear"
    assert laws["grid_plateau"]["params"]["a"] == pytest.approx(MU_0, abs=1e-4)
    assert laws["grid_plateau"]["params"]["b"] == pytest.approx(KAPPA_V, abs=1e-4)
    assert laws["table_size"]["law_form"] == "linear"
    assert laws["table_size"]["params"]["b"] == pytest.approx(KAPPA_M, abs=1e-6)
    assert laws["density"]["law_form"] == "slanted_hyperbolic"
    assert laws["density"]["established_form"] == "two_slope_hyperbolic"
    assert laws["density"]["params"]["mu"] == pytest.approx(P_LAW[0], abs=1e-3)
    assert laws["rule_position"]["law_form"] == "slanted_hyperbolic"
    assert laws["rule_position"]["params"]["kappa"] == pytest.approx(R_LAW[1], abs=1e-3)
    # Four candidates: the simple hyperbola is none, its pole lies at the zero of the abscissa.
    assert len(laws["density"]["candidates"]) == 4
    lambda_r = _hyperbola(R_LAW, 0.0)
    # The slanted hyperbola has the horizontal asymptote μ at p → 0: it is the shift of the density.
    expected = MU_0 + P_LAW[0] + LAMBDA_M + lambda_r
    assert section["alpha"]["value"] == pytest.approx(expected, abs=1e-3)
    by_key = {(row["key"], row["level"], row["method"]): row for row in section["capacities"]}
    # At the level 0.5 the threshold of the logit of the median request is zero at any variance.
    threshold = 0.0
    table, rule = _law(LINE, np.array([LAMBDA_M, KAPPA_M])), _law(RULE_FORM, R_LAW)
    instruction = by_key[("instruction_capacity", 0.5, "law")]["value"]
    root = log_capacity(table, table.params, rule, rule.params, expected - threshold)
    assert instruction == pytest.approx(2.0**root, rel=1e-3)
    # By the bins and by the laws: the coefficients lie on the laws, so the two are close.
    assert by_key[("instruction_capacity", 0.5, "bins")]["value"] == pytest.approx(
        instruction, rel=0.15
    )
    fatigue = by_key[("fatigue_capacity_words", 0.5, "law")]["value"]
    x = math.log2(fatigue)
    logit = expected + KAPPA_V * (x - math.log2(20.0)) - A * 2.0 ** (GAMMA * x)
    assert logit == pytest.approx(threshold, abs=1e-3)


def test_full_density_multiplies_the_odds_of_an_error_by_the_loss_of_its_law() -> None:
    design, fit = _known_fit()
    section = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)
    odds = next(row for row in section["capacities"] if row["key"] == "density_odds_factor")
    loss = section["density_loss_at_one"]
    # The loss goes from the shift of the density, the asymptote μ of its slanted hyperbola at
    # p → 0, to the law at p = 1.
    expected = _hyperbola(P_LAW, 10.0) - P_LAW[0]
    assert -loss == pytest.approx(expected, abs=1e-3)
    # The factor keeps the sign: e^(−P̂(1)) with P̂(1) = −loss.
    assert odds["value"] == pytest.approx(math.exp(loss))
    assert odds["status"] == "measured"
    assert odds["interval"][0] < odds["value"] < odds["interval"][1]


def test_rows_keep_the_reference_bins_and_the_slices() -> None:
    design, fit = _known_fit()
    section = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)
    m_rows = {row["name"]: row for row in section["bins"]["M"]}
    assert m_rows["m_bin_1"]["status"] == "reference"
    assert m_rows["m_bin_1"]["restored"] == pytest.approx(-LAMBDA_M, abs=1e-6)
    assert m_rows["m_bin_1"]["own_half"] > 0.0
    p_rows = {row["name"]: row for row in section["bins"]["P"]}
    # A bin of few requests lies on the law and is not an outlier: it enters the law.
    assert p_rows["p_bin_1"]["status"] == "law"
    assert len(section["bins"]["slices"]) == 8
    assert section["bins"]["slices"][0]["mu_v"] == pytest.approx(MU_0 + KAPPA_V, abs=1e-4)


def test_the_composition_of_the_capacities() -> None:
    design, fit = _known_fit()
    section = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)
    keys = {(row["key"], row["level"], row["method"]) for row in section["capacities"]}
    assert keys == {
        ("instruction_capacity", 0.5, "law"),
        ("instruction_capacity", 0.5, "bins"),
        ("fatigue_capacity_words", 0.5, "law"),
        ("density_odds_factor", None, "law"),
    }


def test_a_corner_off_its_bounds_is_free() -> None:
    design, fit = _known_fit()
    laws = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)["laws"]
    assert laws["density"]["held"] == []
    assert laws["rule_position"]["held"] == []


def test_the_curve_takes_the_variances_and_the_errors_take_the_full_covariance() -> None:
    plain = rule_word_laws(*_known_fit(), 10240.0, ASSIGNMENT)
    common = rule_word_laws(*_known_fit(common_error=0.01), 10240.0, ASSIGNMENT)
    # A common error of all the cells does not move the curve: the weights stay equal.
    assert common["laws"]["grid"]["params"]["A"] == pytest.approx(plain["laws"]["grid"]["params"]["A"], abs=1e-4)
    assert common["laws"]["grid"]["params"]["gamma"] == pytest.approx(
        plain["laws"]["grid"]["params"]["gamma"], abs=1e-4
    )
    # It goes into the error of the level whole: the row of mu_0 sums to one, 1.96 × 0.1.
    assert common["shifts"]["mu_0"]["half"] >= 0.19 > plain["shifts"]["mu_0"]["half"]


def test_gamma_on_a_bound_is_named_and_has_no_error() -> None:
    design, fit = _known_fit()
    names = list(fit.names)
    theta = list(fit.theta)
    # A straight line over log₂ s has no exponent: the search goes to the lower bound of gamma.
    for cell in design.cells:
        x = math.log2(design.abscissae[cell.name]["x"])
        theta[names.index(cell.name)] = MU_0 + KAPPA_V * design.abscissae[cell.name]["u"] - 0.5 * x
    grid = rule_word_laws(design, replace(fit, theta=tuple(theta)), 10240.0, ASSIGNMENT)["laws"]["grid"]
    assert grid["held"] == ["gamma"]
    assert grid["params"]["gamma"] == pytest.approx(0.005, abs=1e-5)
    assert grid["half"]["gamma"] == 0.0


def _with_spread(fit: VariableFit) -> VariableFit:
    single = SpreadAxis(((1,),), (None,), (1.0,), (1.0,), (40,))
    length = SpreadAxis(((1,), (2,)), (None, 1), (100.0, 4000.0), (0.5, 0.5), (40, 40))
    size = len(fit.names)
    return replace(
        fit,
        method="glmm_heterogeneous",
        spread=LoadSpread((0.5 * math.log(2.0), 0.4), (length, single, single)),
        spread_covariance=tuple(tuple(row) for row in (np.eye(size + 2) * 1e-4).tolist()),
    )


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


def test_the_capacities_of_h_do_not_depend_on_the_spread() -> None:
    design, fit = _known_fit()
    plain = rule_word_laws(design, fit, 10240.0, ASSIGNMENT)
    lengths = rule_word_laws(design, _with_spread(fit), 10240.0, ASSIGNMENT)
    rules = rule_word_laws(design, _with_rules_spread(fit), 10240.0, ASSIGNMENT)
    for section in (lengths, rules):
        assert section["capacities"] == plain["capacities"]
        assert all("sigma2" not in row and "threshold" not in row for row in section["capacities"])
        assert "thresholds" not in section
    # Two groups of an axis are too few for an approximation of its variance.
    assert rules["spread_laws"] == {}
