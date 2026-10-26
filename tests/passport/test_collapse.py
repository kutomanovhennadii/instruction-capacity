"""The collapse of the answer: its rate and its place - bins, laws, shifts, checks, figures."""

import json
from pathlib import Path

import numpy as np

from iec.passport.approximation import plateau_level
from iec.passport.bin_layout import BinLayout, doubling_bounds
from iec.passport.collapse import (
    INTERCEPT,
    PLACE_LAW_PARAMETERS,
    RATE_LAW_PARAMETERS,
    SHARE_BINS,
    binned_fit,
    collapse_summary,
    place_summary,
)
from iec.passport.document.figures import (
    COLLAPSE_FIGURE_NAMES,
    PLACE_FIGURE_NAMES,
    write_passport_figures,
)
from iec.passport.document.sections import collapse_lines
from iec.passport.vspmr_model import doub_bin

#: The rate law of the synthetic points: λ_ψ, κ_ψ, A_ψ, γ_ψ.
_LAW = (-6.0, 1.0, 4.0, 0.6)
#: The place law of the synthetic trials: λ_F, κ_SF, log₂ s_*, ω_SF, A_F, γ_F, κ_QF.
_PLACE = (-4.0, 1.2, 5.0, 0.8, 3.0, 0.5, 2.0)


def _layout() -> BinLayout:
    """Eight length bins from 20 words, ten density bins down to 1/512, three rule bins."""
    return BinLayout(
        volume_bounds=doubling_bounds(20, 2560),
        density_bounds=doubling_bounds(1, 512),
        row_bounds=doubling_bounds(1, 4),
        density_reference=3,
    )


def _rate_block(size: int, seed: int) -> dict:
    """The rate of synthetic points by ``_LAW``, through JSON as the passport keeps it.

    A point has the length ``20·2^k`` words (``k = 0…7``), the density ``2^−j`` (``j = 0…9``) and
    one to four rules; its bins are those of ``_layout``.
    """
    rng = np.random.default_rng(seed)
    doublings = rng.integers(0, 8, size)
    sparseness = rng.integers(0, 10, size)
    density = 2.0 ** -sparseness.astype(float)
    eta = _LAW[0] + _LAW[1] * doublings + _LAW[2] * density ** _LAW[3]
    lost = (rng.random(size) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    block = collapse_summary(
        length_words=20.0 * 2.0 ** doublings.astype(float),
        density=density,
        bins={
            "v_bin_": doublings + 1,
            "p_bin_": 10 - sparseness,
            "m_bin_": rng.integers(1, 4, size),
        },
        lost=lost,
        weights=np.ones(size),
        layout=_layout(),
        weighting=None,
    )
    return json.loads(json.dumps(block))


def _place_block(points: int, seed: int) -> dict:
    """The place of synthetic trials by ``_PLACE``, through JSON as the passport keeps it.

    A point has the length ``20·2^k`` words (``k = 3…7``) and the density ``2^−j``
    (``j = 0…9``); every word of its text is a trial with an independent outcome of the law.
    """
    rng = np.random.default_rng(seed)
    level, rise, corner, width, amplitude, gamma, slope = _PLACE
    columns: dict[str, list] = {
        key: [] for key in ("s", "share", "p", "y", "g", "sb", "pb", "qb", "m")
    }
    for index in range(points):
        length = 20 * 2 ** int(rng.integers(3, 8))
        sparseness = int(rng.integers(0, 10))
        rules = int(rng.integers(1, 4))
        s = np.arange(1, length + 1)
        share = (s - 1) / length
        eta = (
            level
            + plateau_level(0.0, rise, corner, width, np.log2(s))
            + amplitude * (2.0**-sparseness) ** gamma
            + slope * share
        )
        columns["s"] += s.tolist()
        columns["share"] += share.tolist()
        columns["p"] += [2.0**-sparseness] * length
        columns["y"] += (rng.random(length) < 1.0 / (1.0 + np.exp(-eta))).astype(float).tolist()
        columns["g"] += [index] * length
        columns["sb"] += [doub_bin(-(-int(value) // 10)) for value in s]
        columns["pb"] += [10 - sparseness] * length
        columns["qb"] += np.minimum(SHARE_BINS * (s - 1) // length, SHARE_BINS - 1).tolist()
        columns["m"] += [rules] * length
    arrays = {key: np.asarray(value) for key, value in columns.items()}
    block = place_summary(
        word=arrays["s"].astype(float),
        share=arrays["share"].astype(float),
        density=arrays["p"].astype(float),
        bins={
            "s_bin_": arrays["sb"].astype(int),
            "p_bin_": arrays["pb"].astype(int),
            "q_bin_": arrays["qb"].astype(int) + 1,
            "v_bin_": np.ones(len(arrays["s"]), dtype=int),
            "m_bin_": arrays["m"].astype(int),
        },
        lost=arrays["y"].astype(float),
        groups=arrays["g"].astype(int),
        point_weights=np.ones(points),
        layout=_layout(),
        weighting=None,
    )
    return json.loads(json.dumps(block))


def test_the_rate_law_recovers_its_parameters() -> None:
    rate = _rate_block(20000, 7)["rate_law"]
    assert tuple(rate["theta"]) == RATE_LAW_PARAMETERS
    for name, truth in zip(RATE_LAW_PARAMETERS, _LAW, strict=True):
        assert abs(rate["theta"][name] - truth) < 4.0 * rate["se"][name]


def test_the_shift_of_an_axis_is_its_law_at_the_zero_and_goes_into_the_level() -> None:
    block = _rate_block(6000, 3)
    laws, shifts, fit = block["laws"], block["shifts"], block["fit"]
    assert shifts["collapse_length"]["value"] == laws["collapse_length"]["lambda_Vpsi"]
    assert shifts["collapse_density"]["value"] == laws["collapse_density"]["mu_Ppsi"]
    level = (
        fit["theta"][INTERCEPT]
        + shifts["collapse_length"]["value"]
        + shifts["collapse_density"]["value"]
    )
    assert abs(block["rate_law"]["theta"]["lambda_psi"] - level) < 1e-12
    for row in block["restored"]["length_profile"]:
        name = f"v_bin_{row['bin']}"
        if name in fit["theta"]:
            raw = fit["theta"][name]
            assert abs(row["value"] - (raw - shifts["collapse_length"]["value"])) < 1e-12
    reference = fit["reference"]["p_bin_"]
    row = block["restored"]["density_profile"][reference - 1]
    assert abs(row["value"] + shifts["collapse_density"]["value"]) < 1e-12


def test_a_bin_with_one_outcome_goes_out_and_the_reference_is_the_next_one() -> None:
    """All points of the shortest length bin kept the text: it has no estimate and goes out."""
    rng = np.random.default_rng(5)
    size = 6000
    length = rng.integers(1, 5, size)
    density = rng.integers(1, 4, size)
    eta = -0.5 + (length - 2) * 1.0 + (density - 2) * 1.0
    lost = (rng.random(size) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    lost[length == 1] = 0.0
    fit = binned_fit(
        {"v_bin_": length, "p_bin_": density},
        {"v_bin_": None, "p_bin_": 2},
        lost,
        np.arange(size),
        np.ones(size),
    )
    assert fit["dropped"] == [
        {"bin": "v_bin_1", "trials": int((length == 1).sum()), "outcome": "kept"}
    ]
    assert fit["reference"] == {"v_bin_": 2, "p_bin_": 2}
    assert fit["names"] == [INTERCEPT, "v_bin_3", "v_bin_4", "p_bin_1", "p_bin_3"]
    index = fit["names"].index("v_bin_4")
    se = float(np.sqrt(fit["covariance"][index, index]))
    assert abs(fit["theta"][index] - 2.0) < 4.0 * se


def test_the_rules_check_has_a_row_per_rule_bin_and_a_wald_statistic() -> None:
    check = _rate_block(6000, 3)["rules_check"]
    rows = check["bins"]
    assert [row["bin"] for row in rows] == [1, 2, 3]
    assert rows[0]["value"] == 0.0 and rows[0]["se"] is None
    for row in rows[1:]:
        assert abs(row["value"]) < 4.0 * row["se"]
    assert check["wald"]["df"] == 2 and 0.0 <= check["wald"]["p"] <= 1.0
    assert check["largest_change_se"] < 1.0


def test_the_place_law_recovers_its_parameters_and_the_level_takes_the_shifts() -> None:
    """The law of the place is recovered from its bins to the accuracy of the bins.

    A bin replaces the smooth law by a constant over a doubling of the word number or a tenth
    of the text, and the word number and the share of the text are tied inside one text: the
    shape parameters through the bins are off by several percent, while their SE on hundreds of
    thousands of trials is far smaller. The tolerance of a shape parameter is therefore 4 SE
    plus 15 % of its true value; the level, 4 SE plus half a logit.
    """
    place = _place_block(300, 13)
    law, shifts, fit = place["place_law"], place["shifts"], place["fit"]
    assert tuple(law["theta"]) == PLACE_LAW_PARAMETERS
    truth = dict(zip(PLACE_LAW_PARAMETERS, _PLACE, strict=True))
    truth["s_star_F"] = 2.0 ** _PLACE[2]
    for name in ("kappa_SF", "A_F", "gamma_F", "kappa_QF"):
        tolerance = 4.0 * law["se"][name] + 0.15 * abs(truth[name])
        assert abs(law["theta"][name] - truth[name]) < tolerance, name
    tolerance = 4.0 * law["se"]["lambda_F"] + 0.5
    assert abs(law["theta"]["lambda_F"] - truth["lambda_F"]) < tolerance
    level = fit["theta"][INTERCEPT] + sum(shift["value"] for shift in shifts.values())
    assert abs(law["theta"]["lambda_F"] - level) < 1e-12
    assert shifts["place_word"]["value"] == place["laws"]["place_word"]["mu_SF"]
    assert shifts["place_share"]["value"] == place["laws"]["place_share"]["lambda_QF"]


def test_a_held_corner_of_the_word_law_has_no_error() -> None:
    place = _place_block(300, 17)
    law = place["laws"]["place_word"]
    for name in law["held"]:
        assert law["se"][name] is None
        assert place["place_law"]["se"][name] is None


def test_the_collapse_figures_and_section_are_written_from_the_block(tmp_path: Path) -> None:
    block = _rate_block(4000, 11)
    block["place"] = _place_block(300, 17)
    passport = {"model": {"id": "mock-model", "experiment_id": "demo"}, "collapse": block}
    written = write_passport_figures(passport, tmp_path, "mock-model")
    assert list(written) == [*COLLAPSE_FIGURE_NAMES, *PLACE_FIGURE_NAMES]
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0
    text = "\n".join(collapse_lines(passport, written))
    assert r"\lambda_{V\psi}" in text and r"\mu_{SF}" in text
    for name in (*COLLAPSE_FIGURE_NAMES, *PLACE_FIGURE_NAMES):
        assert f"/{name})" in text


def test_a_passport_without_the_block_has_no_collapse_figures_nor_section(tmp_path: Path) -> None:
    passport = {"model": {"id": "mock-model", "experiment_id": "demo"}, "collapse": None}
    assert write_passport_figures(passport, tmp_path, "mock-model") == {}
    assert collapse_lines(passport, {}) == []
