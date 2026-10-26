"""Tests of the approximation of the VS-PMR passport."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus, TrialRecord
from iec.passport.approximation import (
    STATUS_LAW_NOT_MEASURED,
    STATUS_PARTIAL,
    _fit_plateau_law,
    _plateau_law_curve,
    _plateau_law_jacobian,
    _vs_curve,
    approximation_block,
    bin_abscissae,
    component_correlations,
    fit_vs_law,
    plateau_level,
    vs_law_value,
)
from iec.passport.bin_layout import BinLayout, doubling_bounds
from iec.passport.glmm_fit import marginal_threshold
from iec.passport.logit_fit import INFORMATION_RCOND
from iec.passport.measured_points import MeasuredPoint
from iec.passport.second_stage_laws import (
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    delta_se,
    increment_matrix,
)
from iec.passport.vspmr_model import (
    VspmrGlmmModel,
    vs_grid_cells,
    vspmr_density_reference,
    vspmr_density_p,
)

#: The surface of the synthetic data: ``κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V``.
_SURFACE = np.array([-2.0, 6.0, 3.0, -1.5, 1.0, 1.5, 0.5])


def _layout_of(volume_top: int, density_top: int, rows_top: int) -> BinLayout:
    """Length from 20 words, density and rules from one; the reference density bin holds 4."""
    return BinLayout(
        volume_bounds=doubling_bounds(20, volume_top),
        density_bounds=doubling_bounds(1, density_top),
        row_bounds=doubling_bounds(1, rows_top),
        density_reference=3,
    )


def _layout() -> BinLayout:
    """Length 1-20, 21-40, 41-80; density up to 16 (reference bin 3); rules up to 8."""
    return _layout_of(80, 16, 8)


def _wide_layout() -> BinLayout:
    """Five length bins (1-20 … 161-320 words) - the plateau hyperbola has four parameters and
    needs five lengths; rules up to 32 - six R bins; twenty grid cells, seven density bins with
    the fifth (VS-PMR numbering) as the reference."""
    return _layout_of(320, 64, 32)


def _point(
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    positions: tuple[int, ...],
    *,
    status: PointStatus = PointStatus.OK,
    seed: int = 1,
) -> MeasuredPoint:
    trials = tuple(
        TrialRecord(
            t=index,
            rule_index=1,
            word_position=position,
            expected="a",
            observed="a",
            z=1,
            outcome="correct",
        )
        for index, position in enumerate(positions, start=1)
    )
    record = PointRecord(
        conditions=PointConditions(D=D, n=n, m=m, seed_rules=seed, seed_text=2, seed_order=0),
        rules=(),
        source_text="",
        answer="",
        status=status,
        prompt_tokens=64,
        response_tokens=20,
    )
    return MeasuredPoint(record=record, trials=trials, first_fail_kind_3_line=None)


def test_the_cell_centre_is_the_geometric_mean_of_word_positions() -> None:
    """The word position is the abscissa itself (the first word is 1); the centre is by ``log₂ s``."""
    centres = bin_abscissae([_point(4, 5, 1, (1, 4, 2))], _layout(), None, 20)
    assert centres["vs_1_1"]["x"] == pytest.approx(2.0)  # 2^((0 + 2 + 1)/3)


def test_centres_are_weighted_means_of_the_trials() -> None:
    """Two points of one cell with weights 1 and 3: the means are computed by hand."""
    points = [_point(5, 5, 1, (5, 15), seed=1), _point(6, 5, 1, (5, 15), seed=2)]
    centres = bin_abscissae(points, _layout(), {points[0].key: 1.0, points[1].key: 3.0}, 20)

    u_expected = (math.log2(25 / 20) + 3.0 * math.log2(30 / 20)) / 4.0
    assert centres["vs_2_1"]["x"] == pytest.approx(5.0)
    assert centres["vs_2_1"]["u"] == pytest.approx(u_expected)
    assert centres["vs_2_2"]["x"] == pytest.approx(15.0)
    assert centres["vs_2_2"]["u"] == pytest.approx(u_expected)
    # p = 1/5 and 1/6 lie in one density bin (level -3).
    p = [value["p"] for name, value in centres.items() if name.startswith("p_bin_")]
    assert p == [pytest.approx((0.2 + 3.0 / 6.0) / 4.0)]
    assert centres["m_bin_1"]["m"] == pytest.approx(1.0)
    assert centres["r_bin_1"]["r"] == pytest.approx(1.0)


def test_a_point_that_is_not_ok_does_not_count() -> None:
    points = [
        _point(5, 5, 1, (5,)),
        _point(6, 5, 1, (5,), status=PointStatus.API_ERROR, seed=2),
    ]
    centres = bin_abscissae(points, _layout(), None, 20)
    assert centres["vs_2_1"]["u"] == pytest.approx(math.log2(25 / 20))


def test_a_cell_without_trials_is_absent() -> None:
    centres = bin_abscissae([_point(5, 5, 1, (5,))], _layout(), None, 20)
    assert "vs_3_1" not in centres
    assert "vs_2_2" not in centres


def test_a_position_past_the_bin_falls_into_its_last_cell() -> None:
    """``v = 20`` is the first length bin, its last cell is the second; position 30 is line 3."""
    centres = bin_abscissae([_point(4, 5, 1, (30,))], _layout(), None, 20)
    assert set(name for name in centres if name.startswith("vs_")) == {"vs_1_2"}
    assert centres["vs_1_2"]["x"] == pytest.approx(30.0)
    assert centres["vs_1_2"]["u"] == pytest.approx(0.0)


# --- The approximation block -------------------------------------------------------------


#: Laws of the synthetic data.
_P_LAW = (1.0, 3.0, 0.3)  # mu_P, A_P, p_c
_M_LAW = (0.2, 0.4 * math.log(2.0))  # lambda_M, kappa_M - the earlier line 0.2 + 0.4·ln m
_R_LAW = (0.05, 0.05, 1.2)  # lambda_R, A_R, gamma - the earlier curve 0.1 − 0.05·r^1.2


def _synthetic(layout, drop: tuple[str, ...] = ()) -> tuple[dict[str, float], dict]:
    """Build ``θ`` exactly on the laws at the centres of mass, and the centres (reference included)."""
    names = [name for name in VspmrGlmmModel().parameter_names(layout) if name not in drop]
    abscissae: dict[str, dict[str, float]] = {}
    theta: dict[str, float] = {}
    for volume, position in vs_grid_cells(layout):
        coords = {"x": 2.0 ** (position - 0.25), "u": (volume - 1) - 0.1}
        abscissae[f"vs_{volume}_{position}"] = coords
    for number in range(1, layout.P + 1):
        low, high = vspmr_density_p(layout, number)
        abscissae[f"p_bin_{number}"] = {"p": math.sqrt(low * high)}
    for number in range(2, layout.R + 1):
        low, high = layout.row_bounds[number - 1]
        centre = math.sqrt(low * high)
        abscissae[f"m_bin_{number}"] = {"m": centre}
        abscissae[f"r_bin_{number}"] = {"r": centre}
    mu_p, a_p, p_c = _P_LAW
    lambda_m, kappa_m = _M_LAW
    lambda_r, a_r, gamma = _R_LAW
    for name in names:
        coords = abscissae[name]
        if name.startswith("vs_"):
            point = np.array([[math.log2(coords["x"]), coords["u"]]])
            theta[name] = float(_vs_curve(_SURFACE, point)[0])
        elif name.startswith("p_bin_"):
            theta[name] = mu_p - a_p * (1.0 - math.exp(-coords["p"] / p_c))
        elif name.startswith("m_bin_"):
            theta[name] = lambda_m + kappa_m * math.log2(coords["m"])
        else:
            theta[name] = lambda_r - a_r * (coords["r"] ** gamma - 1.0)
    return theta, abscissae


def test_the_block_recovers_all_four_laws() -> None:
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    laws = block["laws"]
    assert block["status"] == STATUS_MEASURED
    grid = laws["vs_grid"]
    # The coding with the shift: δ_VS is the horizontal asymptote μ₀ of the plateau; λ_S
    # moves by it, κ_S, ω_S, κ_V and the corner do not; the corner is a length in words.
    assert "vs_grid_by_length" not in block
    assert grid["delta_VS"] == pytest.approx(_SURFACE[2], abs=1e-4)
    assert [grid[k] for k in ("kappa_S", "kappa_V", "omega_S", "omega_V")] == pytest.approx(
        list(_SURFACE[[0, 3, 4, 6]]), abs=1e-4
    )
    assert grid["v_star"] == pytest.approx(20.0 * 2.0 ** _SURFACE[5], rel=1e-4)
    assert grid["lambda_S"] + grid["delta_VS"] == pytest.approx(_SURFACE[1], abs=1e-4)
    assert "mu_0" not in grid and "u_star" not in grid
    assert grid["held"] == []
    assert [row["determined"] for row in grid["slices"]] == [True] * 5
    assert [laws["density"][k] for k in ("mu_P", "A_P", "p_c")] == pytest.approx(
        list(_P_LAW), abs=1e-4
    )
    assert [laws["table_size"][k] for k in ("lambda_M", "kappa_M")] == pytest.approx(
        list(_M_LAW), abs=1e-4
    )
    assert [laws["rule_position"][k] for k in ("lambda_R", "A_R", "gamma")] == pytest.approx(
        list(_R_LAW), abs=1e-4
    )
    assert all("sensitivity" not in law for law in laws.values())
    assert set(grid["se"]) == {
        "delta_VS", "kappa_S", "lambda_S", "omega_S", "kappa_V", "v_star", "omega_V"
    }


def test_the_block_publishes_public_quantities() -> None:
    """Public quantities: keys and order, the formulas of the values."""
    from iec.passport.public_quantities import PUBLIC_KEYS

    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    public = block["public"]
    rows = {(row["key"], row["level"], row["method"]): row for row in public}
    mu_p, a_p, p_c = _P_LAW
    full = rows[("loss_at_full_density", None, None)]
    assert full["value"] == pytest.approx(a_p * (1.0 - math.exp(-1.0 / p_c)), abs=1e-4)
    assert full["se"] > 0.0
    assert rows[("volume_price_per_doubling", None, None)]["value"] == pytest.approx(
        -_SURFACE[3], abs=1e-4
    )
    assert rows[("fatigue_price_per_doubling", None, None)]["value"] == pytest.approx(
        -_SURFACE[0], abs=1e-4
    )
    assert rows[("rule_set_discount_per_doubling", None, None)]["value"] == pytest.approx(
        _M_LAW[1], abs=1e-4
    )
    present = [key for key in PUBLIC_KEYS if any(row["key"] == key for row in public)]
    assert present == list(PUBLIC_KEYS)
    assert not any(row["key"].startswith("safe_density") for row in public)
    assert not any(row["key"] == "fatigue_reserve_words" for row in public)
    assert not any(row["key"] == "volume_capacity_words" for row in public)
    # IEC and IGC by two calculations at every level; the fatigue capacity by the asymptote only.
    for level in (0.95, 0.5):
        for method in ("bins", "law"):
            assert ("iec_capacity", level, method) in rows
            assert ("igc_capacity", level, method) in rows
        assert ("fatigue_capacity_words", level, "law") in rows
        assert ("fatigue_capacity_words", level, "bins") not in rows
    # The corner and the position to the level by length slices are not published (X12).
    assert "slices_to_level" not in block


def test_the_fatigue_capacity_by_law_crosses_the_slanted_asymptote() -> None:
    """By the law: ``log₂s = (logit q − α_phys − λ_S)/κ_S`` in the coding with the shift
    (``λ_S`` published)."""
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    grid = block["laws"]["vs_grid"]
    alpha = block["alpha_phys"]["value"]
    rows = [
        row
        for row in block["public"]
        if row["key"] == "fatigue_capacity_words" and row["method"] == "law"
    ]
    assert len(rows) == 2
    for row in rows:
        q = row["level"]
        numerator = math.log(q / (1.0 - q)) - alpha - grid["lambda_S"]
        if numerator >= 0.0:
            assert row["value"] is None
            assert row["status"] == "below_level_at_first_word"
            continue
        assert row["value"] == pytest.approx(2.0 ** (numerator / grid["kappa_S"]), rel=1e-6)
        assert row["status"] in {"measured", "extrapolated"}


def test_the_igc_by_law_adds_the_full_density_loss_to_the_cumulative_price() -> None:
    """By the laws ``C(m) + A_P·(1 − e^(−1/p_c)) = Δ_q`` holds at the IGC; it is below the IEC.

    In the synthetic data the reserve of ``q = 0.95`` is below the loss at full density - the
    level is not held even with one rule; the reserve of ``q = 0.5`` is above it.
    """
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    laws = block["laws"]
    kappa_m = laws["table_size"]["kappa_M"]
    a_r, gamma = laws["rule_position"]["A_R"], laws["rule_position"]["gamma"]
    loss = laws["density"]["A_P"] * (1.0 - math.exp(-1.0 / laws["density"]["p_c"]))
    alpha = block["alpha_phys"]["value"]
    rows = {(row["key"], row["level"], row["method"]): row for row in block["public"]}
    assert rows[("igc_capacity", 0.5, "law")]["status"] == "measured"
    for q in (0.95, 0.5):
        reserve = alpha - math.log(q / (1.0 - q))
        row = rows[("igc_capacity", q, "law")]
        if reserve <= loss:
            assert row["value"] is None
            assert row["status"] == "reserve_not_positive"
            continue
        m = row["value"]
        cost = -kappa_m * math.log2(m) + a_r * (
            (m ** (gamma + 1.0) - 1.0) / (gamma + 1.0) - (m - 1.0)
        )
        assert cost + loss == pytest.approx(reserve, abs=1e-6)
        assert m < rows[("iec_capacity", q, "law")]["value"]


def test_the_igc_by_bins_walks_the_rule_prices_with_the_restored_full_density_bin() -> None:
    """By the bins: the IEC walk with the reserve ``Δ_q + P(1)``, ``P(1)`` the restored bin
    ``p = 1``; the IGC is below the IEC."""
    from iec.passport.public_quantities import iec_capacity, rule_price_support_max
    from iec.passport.vspmr_model import rule_price

    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    full = theta[f"p_bin_{layout.P}"] - block["laws"]["density"]["mu_P"]
    r_shift = block["shifts"]["rule_position"]["value"]
    support = rule_price_support_max(layout)
    prices = [rule_price(r, theta, layout) + r_shift for r in range(2, support + 1)]
    alpha = block["alpha_phys"]["value"]
    rows = {(row["key"], row["level"], row["method"]): row for row in block["public"]}
    assert rows[("igc_capacity", 0.5, "bins")]["status"] == "measured"
    for q in (0.95, 0.5):
        reserve = alpha - math.log(q / (1.0 - q)) + full
        row = rows[("igc_capacity", q, "bins")]
        if reserve <= 0.0:
            assert row["value"] is None
            assert row["status"] == "reserve_not_positive"
            continue
        expected = iec_capacity(reserve, prices, support).fractional
        assert row["value"] == pytest.approx(expected, rel=1e-9)
        assert row["value"] < rows[("iec_capacity", q, "bins")]["value"]


def test_a_spread_moves_the_level_to_its_threshold() -> None:
    """With the shift inside the fit the level is read at η*(q, σ²), and the error of σ² enters.

    The fatigue capacity has a closed form: ``log₂s = (η* − α_phys − λ_S)/κ_S`` in the coding
    with the shift. The same ``σ²`` without its variance gives the same values and a smaller
    SE; at ``q = 0.5`` the threshold does not move with ``σ²``, and the SE is the same.
    """
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    covariance = 0.01 * np.eye(len(theta))
    plain = approximation_block(
        theta,
        covariance,
        layout,
        abscissae,
        reference_total_words=20,
        sigma2_in_fit=(1.0, [0.0] * (len(theta) + 1)),
    )
    spread = approximation_block(
        theta,
        covariance,
        layout,
        abscissae,
        reference_total_words=20,
        sigma2_in_fit=(1.0, [0.0] * len(theta) + [0.04]),
    )
    grid = spread["laws"]["vs_grid"]
    alpha = spread["alpha_phys"]["value"]

    def capacity(block: dict[str, Any], q: float) -> dict[str, Any]:
        (row,) = [
            row
            for row in block["public"]
            if row["key"] == "fatigue_capacity_words" and row["level"] == q
        ]
        return row

    for q in (0.95, 0.5):
        numerator = marginal_threshold(q, 1.0)[0] - alpha - grid["lambda_S"]
        expected = 2.0 ** (numerator / grid["kappa_S"])
        assert capacity(spread, q)["value"] == pytest.approx(expected, rel=1e-6)
        assert capacity(spread, q)["value"] == capacity(plain, q)["value"]
    assert capacity(spread, 0.95)["se"] > capacity(plain, 0.95)["se"]
    assert capacity(spread, 0.5)["se"] == pytest.approx(capacity(plain, 0.5)["se"], rel=1e-9)


def test_the_binned_iec_is_fractional_with_a_delta_method_error() -> None:
    """By bins: a fractional capacity where the reserve is exhausted inside the support, its SE."""
    from iec.passport.public_quantities import _binned_iec

    layout = _wide_layout()
    theta, _ = _synthetic(layout)
    # Rule prices grow: every M bin is 0.5 below the previous one, R - 0.2.
    for k in range(2, layout.R + 1):
        theta[f"m_bin_{k}"] = -0.5 * (k - 1)
        theta[f"r_bin_{k}"] = -0.2 * (k - 1)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    row = _binned_iec(theta, names, covariance, layout, 3.0, np.zeros(len(names)), 32, 0.95, "0_95")
    assert row["status"] == STATUS_MEASURED
    assert row["value"] != int(row["value"])
    assert row["se"] is not None and row["se"] > 0.0
    # Root check: the cumulative price up to the fractional capacity equals the reserve.
    from iec.passport.vspmr_model import rule_price

    m_last = int(row["value"])
    cumulative = sum(rule_price(r, theta, layout) for r in range(2, m_last + 1))
    price = rule_price(m_last + 1, theta, layout)
    assert cumulative + (row["value"] - m_last) * price == pytest.approx(3.0, abs=1e-9)


def test_the_binned_iec_prices_the_restored_bins() -> None:
    """Rule prices by the restored bins: the raw price plus the shift of the ``R`` block.

    The resource that early rules add (positive restored bins) must enter the prices, not be
    subtracted from the reserve.
    """
    from iec.passport.public_quantities import _binned_iec
    from iec.passport.vspmr_model import rule_price

    layout = _wide_layout()
    theta, _ = _synthetic(layout)
    for k in range(2, layout.R + 1):
        theta[f"m_bin_{k}"] = -0.5 * (k - 1)
        theta[f"r_bin_{k}"] = -0.2 * (k - 1)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    shift = 0.1
    shift_gradient = np.zeros(len(names))
    shift_gradient[names.index("r_bin_2")] = 1.0
    raw = _binned_iec(theta, names, covariance, layout, 3.0, np.zeros(len(names)), 32, 0.95, "0_95")
    restored = _binned_iec(
        theta, names, covariance, layout, 3.0, np.zeros(len(names)), 32, 0.95, "0_95",
        price_shift=(shift, shift_gradient),
    )
    # Every price is dearer by the shift - the reserve ends earlier.
    assert restored["value"] < raw["value"]
    m_last = int(restored["value"])
    cumulative = sum(rule_price(r, theta, layout) + shift for r in range(2, m_last + 1))
    price = rule_price(m_last + 1, theta, layout) + shift
    assert cumulative + (restored["value"] - m_last) * price == pytest.approx(3.0, abs=1e-9)
    assert restored["se"] is not None and restored["se"] != raw["se"]


def test_restored_bins_keep_every_prediction() -> None:
    """Invariant: a restored bin plus the block shift is the raw coefficient."""
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    shifts = {key: value["value"] for key, value in block["shifts"].items()}
    assert sum(shifts.values()) == pytest.approx(block["alpha_phys"]["value"], abs=1e-12)
    assert shifts["vs_grid"] == block["laws"]["vs_grid"]["delta_VS"]
    assert shifts["density"] == block["laws"]["density"]["mu_P"]
    assert shifts["table_size"] == block["laws"]["table_size"]["lambda_M"]
    assert shifts["rule_position"] == block["laws"]["rule_position"]["lambda_R"]

    restored = block["restored"]
    reference = vspmr_density_reference(layout)
    for row in restored["vs_grid"]:
        name = f"vs_{row['v_bin']}_{row['s_bin']}"
        assert row["value"] + shifts["vs_grid"] == pytest.approx(theta[name], abs=1e-10)
    for row in restored["density_profile"]:
        raw = 0.0 if row["bin"] == reference else theta[f"p_bin_{row['bin']}"]
        assert row["value"] + shifts["density"] == pytest.approx(raw, abs=1e-10)
    for key, prefix, shift in (
        ("table_size_profile", "m_bin_", shifts["table_size"]),
        ("rule_position_profile", "r_bin_", shifts["rule_position"]),
    ):
        for row in restored[key]:
            raw = 0.0 if row["bin"] == 1 else theta[f"{prefix}{row['bin']}"]
            assert row["value"] + shift == pytest.approx(raw, abs=1e-10)


def test_on_exact_laws_the_bins_lie_on_their_curves() -> None:
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    for rows in block["restored"].values():
        for row in rows:
            if row["z"] is None:  # the reference bin is a coding, not a point
                continue
            assert row["z"] == pytest.approx(0.0, abs=1e-6)
            assert row["law"] == pytest.approx(row["value"], abs=1e-6)
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, 20)
    # The law tends to its horizontal asymptote μ₀ = δ_VS at the start of the answer of
    # short texts.
    assert vs_law_value(law, 2.0**-1000, -1000.0) == pytest.approx(law["mu_0"], abs=1e-3)
    # Far to the right - the slanted asymptote κ_S·log₂s + λ_S (the hyperbola approaches as
    # κ_S²ω_S²/(4|d|)).
    far = 60.0
    assert vs_law_value(law, 2.0**far, 0.0) == pytest.approx(
        law["kappa_S"] * far + law["lambda_S"], abs=1e-1
    )


def _block_identifiability(
    covariance: np.ndarray,
    names: Sequence[str],
    blocks: Mapping[str, Sequence[str | None]],
) -> dict[str, dict[str, Any]]:
    """Reference: canonical correlation of every pair of blocks."""
    order = list(names)
    increments: dict[str, tuple[list[int], np.ndarray]] = {}
    for key, block in blocks.items():
        members = [order.index(name) for name in block if name is not None]
        increments[key] = (members, increment_matrix(block))
    keys = list(blocks)
    result: dict[str, dict[str, Any]] = {}
    for left_position, left in enumerate(keys):
        for right in keys[left_position + 1 :]:
            rows, a_left = increments[left]
            cols, a_right = increments[right]
            if not rows or not cols:
                continue
            s_aa = a_left @ covariance[np.ix_(rows, rows)] @ a_left.T
            s_bb = a_right @ covariance[np.ix_(cols, cols)] @ a_right.T
            s_ab = a_left @ covariance[np.ix_(rows, cols)] @ a_right.T
            product = (
                np.linalg.pinv(s_aa, rcond=INFORMATION_RCOND, hermitian=True)
                @ s_ab
                @ np.linalg.pinv(s_bb, rcond=INFORMATION_RCOND, hermitian=True)
                @ s_ab.T
            )
            largest = float(np.max(np.real(np.linalg.eigvals(product))))
            result[f"{left}_{right}"] = {
                "canonical_correlation": math.sqrt(min(max(largest, 0.0), 1.0)),
            }
    return result


def test_component_correlations_agree_with_block_identifiability() -> None:
    layout = _wide_layout()
    theta, _ = _synthetic(layout)
    names = list(theta)
    rng = np.random.default_rng(20261007)
    factor = rng.normal(size=(len(names), len(names)))
    covariance = factor @ factor.T / len(names) + 0.01 * np.eye(len(names))

    result = component_correlations(covariance, names, layout)
    canonical = np.array(result["canonical"], dtype=float)
    assert set(result) == {"components", "canonical"}
    assert result["components"] == ["V", "S", "P", "M", "R"]
    assert canonical.shape == (5, 5)
    assert np.allclose(canonical, canonical.T)
    assert np.allclose(np.diag(canonical), 1.0)

    reference = vspmr_density_reference(layout)
    blocks = {
        "P": [None if k == reference else f"p_bin_{k}" for k in range(1, layout.P + 1)],
        "M": [None, *(f"m_bin_{k}" for k in range(2, layout.R + 1))],
        "R": [None, *(f"r_bin_{k}" for k in range(2, layout.R + 1))],
    }
    expected = _block_identifiability(covariance, names, blocks)
    position = {"P": 2, "M": 3, "R": 4}
    for pair, values in expected.items():
        left, right = pair.split("_")
        i, j = position[left], position[right]
        assert canonical[i, j] == pytest.approx(values["canonical_correlation"], abs=1e-10)


def test_independent_coefficients_give_no_cross_correlation_between_blocks() -> None:
    """With a diagonal ``Σ`` different blocks are unrelated; ``V`` and ``S`` are increments of
    one grid and share cells, so their link remains even with independent cells."""
    layout = _wide_layout()
    theta, _ = _synthetic(layout)
    names = list(theta)
    canonical = np.array(
        component_correlations(0.01 * np.eye(len(names)), names, layout)["canonical"],
        dtype=float,
    )
    for i in range(5):
        for j in range(5):
            if i == j or {i, j} == {0, 1}:
                continue
            assert canonical[i, j] == pytest.approx(0.0, abs=1e-10)
    assert canonical[0, 1] > 0.0


def test_three_length_slices_leave_the_block_partial() -> None:
    """Three length slices: the plateau hyperbola (four parameters) is not drawn, the grid law is not measured."""
    layout = _wide_layout()
    drop = (
        *(f"vs_4_{s}" for s in range(1, 6)),
        *(f"vs_5_{s}" for s in range(1, 7)),
        "vs_3_3",
        "vs_3_4",
    )
    theta, abscissae = _synthetic(layout, drop=drop)
    block = approximation_block(
        theta, 0.01 * np.eye(len(theta)), layout, abscissae, reference_total_words=20
    )
    assert block["status"] == STATUS_PARTIAL
    assert block["alpha_phys"] is None
    assert block["shifts"] is None
    assert block["laws"]["vs_grid"]["status"] == STATUS_INSUFFICIENT
    statuses = {
        f"vs_{row['v_bin']}_{row['s_bin']}": row["status"] for row in block["restored"]["vs_grid"]
    }
    assert all(statuses[name] == "excluded" for name in drop)
    assert all(
        status == STATUS_LAW_NOT_MEASURED
        for name, status in statuses.items()
        if name not in drop
    )
    for key in ("density_profile", "table_size_profile", "rule_position_profile"):
        assert all(row["status"] == STATUS_MEASURED for row in block["restored"][key])
    assert np.array(block["component_correlations"]["canonical"], dtype=float).shape == (5, 5)


# --- The grid law by length slices -------------------------------------------------------


def _grid_only(layout, values):
    """``θ`` of the grid cells by a function ``(x, u, slice number) → value``, synthetic centres."""
    _, abscissae = _synthetic(layout)
    theta = {}
    for volume, position in vs_grid_cells(layout):
        coords = abscissae[f"vs_{volume}_{position}"]
        theta[f"vs_{volume}_{position}"] = values(coords["x"], coords["u"], volume)
    return theta, abscissae


def _free_plateaus(plateaus: Mapping[int, float]) -> Any:
    """A surface whose plateau is its own in every length slice: ``a = −2``, ``b = 6``, ``W_S = 1``."""

    def surface(x: float, u: float, volume: int) -> float:
        del u
        params = np.array([-2.0, 6.0, plateaus[volume], 0.0, 1.0, 1.0, 0.5])
        return float(_vs_curve(params, np.array([[math.log2(x), 0.0]]))[0])

    return surface


#: A surface whose plateau is a hyperbola in the length: ``c₀ = 4``, ``c₁ = −1.5``,
#: ``u_V = 1.5``, ``W_V = 0.5``.
_HYPERBOLIC = np.array([-2.0, 6.0, 4.0, -1.5, 1.0, 1.5, 0.5])


def _hyperbolic_surface(x: float, u: float, volume: int) -> float:
    """The surface ``_HYPERBOLIC`` at a cell: the plateau is the hyperbola itself."""
    del volume
    return float(_vs_curve(_HYPERBOLIC, np.array([[math.log2(x), u]]))[0])


def test_the_plateau_by_length_recovers_free_levels() -> None:
    """Free plateaus of six slices: the fit returns each at shared κ_S, λ_S, ω_S; the plateaus are
    published minus the shift δ_VS - the horizontal asymptote of the plateau hyperbola through
    them."""
    layout = _layout_of(640, 64, 32)
    plateaus = {1: 4.0, 2: 3.8, 3: 3.0, 4: 1.0, 5: -0.5, 6: -2.0}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, 20)
    assert law["status"] == STATUS_MEASURED
    bias = law["mu_0"]
    assert [law["kappa_S"], law["lambda_S"], law["omega_S"]] == pytest.approx(
        [-2.0, 6.0, 1.0], abs=1e-5
    )
    assert all(row["determined"] for row in law["slices"])
    assert [row["mu_v"] + bias for row in law["slices"]] == pytest.approx(
        list(plateaus.values()), abs=1e-5
    )
    assert law["chi2"] < 1e-8
    assert [row["cells"] for row in law["slices"]] == [2, 3, 4, 5, 6, 7]
    assert law["plateau_n_points"] == 6


def test_a_slice_whose_cells_all_lie_on_the_slope_is_undetermined() -> None:
    """A slice with its plateau far above its cells: the plateau is not seen, no number."""
    layout = _layout_of(640, 64, 32)
    # The corner of slice 1 is left of its cells.
    plateaus = {1: 40.0, 2: 3.8, 3: 3.0, 4: 1.0, 5: -0.5, 6: -2.0}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, 20)
    assert law["status"] == STATUS_MEASURED
    first, *rest = law["slices"]
    assert first["determined"] is False
    assert "the plateau is not seen" in first["reason"]
    assert all(row["determined"] for row in rest)
    bias = law["mu_0"]
    assert [row["mu_v"] + bias for row in rest] == pytest.approx([3.8, 3.0, 1.0, -0.5, -2.0], abs=1e-4)


def test_fewer_than_three_determined_slices_give_no_plateau() -> None:
    layout = _layout()
    theta, abscissae = _synthetic(layout, drop=("vs_1_1", "vs_1_2"))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, 20)
    assert law["status"] == STATUS_INSUFFICIENT
    assert [row["determined"] for row in law["slices"]] == [False, False]


def test_fewer_than_five_determined_slices_leave_the_zero_undefined() -> None:
    """Four determined slices: the plateau law (four parameters) is not drawn, the zero is undefined."""
    layout = _layout_of(160, 64, 32)
    plateaus = {1: 4.0, 2: 3.0, 3: 1.0, 4: 0.5}
    theta, abscissae = _grid_only(layout, _free_plateaus(plateaus))
    names = list(theta)
    law = fit_vs_law(theta, 0.01 * np.eye(len(names)), names, abscissae, layout, 20)
    assert law["status"] == STATUS_INSUFFICIENT
    assert all(not row["determined"] for row in law["slices"])
    assert all("the plateau law is not drawn" in row["reason"] for row in law["slices"])


def test_on_a_hyperbolic_plateau_the_levels_lie_on_the_law() -> None:
    """The plateau of the surface is the hyperbola itself: the plateaus of the slices lie on
    it, the plateau hyperbola returns it, and the shift is its horizontal asymptote μ₀ = 4."""
    layout = _layout_of(640, 64, 32)
    theta, abscissae = _grid_only(layout, _hyperbolic_surface)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    law = fit_vs_law(theta, covariance, names, abscissae, layout, 20)
    assert law["status"] == STATUS_MEASURED
    assert law["mu_0"] == pytest.approx(4.0, abs=1e-5)
    assert law["lambda_S"] == pytest.approx(6.0, abs=1e-5)
    assert [law[k] for k in ("kappa_V", "u_star", "omega_V")] == pytest.approx(
        [-1.5, 1.5, 0.5], abs=1e-4
    )
    assert law["v_star"] == pytest.approx(20.0 * 2.0**1.5, rel=1e-4)
    for row in law["slices"]:
        assert row["determined"]
        expected = float(plateau_level(4.0, -1.5, 1.5, 0.5, row["u"])) - 4.0
        assert row["mu_v"] == pytest.approx(expected, abs=1e-5)
        assert row["law"] == pytest.approx(row["mu_v"], abs=1e-5)
        assert row["se"] > 0.0
        assert row["se_fit"] > 0.0
    assert law["held"] == []
    assert law["plateau_chi2"] < 1e-8
    assert law["plateau_n_points"] == 6
    assert delta_se(law["sensitivity"][3], covariance) > 0.0


def test_the_grid_law_sensitivity_is_the_first_order_response() -> None:
    """Both steps: moving one cell by δ moves the seven parameters by the sensitivity times δ."""
    layout = _layout_of(640, 64, 32)
    theta, abscissae = _grid_only(layout, _hyperbolic_surface)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    base = fit_vs_law(theta, covariance, names, abscissae, layout, 20)
    delta = 1e-3
    moved = dict(theta)
    moved[names[5]] += delta
    shifted = fit_vs_law(moved, covariance, names, abscissae, layout, 20)
    keys = ("kappa_S", "lambda_S", "mu_0", "kappa_V", "omega_S", "u_star", "omega_V")
    change = np.array([shifted[key] - base[key] for key in keys])
    assert change == pytest.approx(base["sensitivity"][:, 5] * delta, abs=1e-5)


def test_the_published_corner_is_a_length_in_words() -> None:
    """``v_* = 20·2^{u_*}`` with the error ``ln 2·v_*·SE(u_*)``; ``λ_S`` minus ``δ_VS``; the
    slices are published as the fit gives them."""
    layout = _wide_layout()
    theta, abscissae = _synthetic(layout)
    names = list(theta)
    covariance = 0.01 * np.eye(len(names))
    law = fit_vs_law(theta, covariance, names, abscissae, layout, 20)
    block = approximation_block(theta, covariance, layout, abscissae, reference_total_words=20)
    grid = block["laws"]["vs_grid"]
    assert grid["v_star"] == pytest.approx(20.0 * 2.0 ** law["u_star"], rel=1e-12)
    se_u = delta_se(law["sensitivity"][5], covariance)
    assert grid["se"]["v_star"] == pytest.approx(math.log(2.0) * grid["v_star"] * se_u, rel=1e-9)
    assert grid["lambda_S"] == pytest.approx(law["lambda_S"] - law["mu_0"], abs=1e-12)
    assert grid["delta_VS"] == law["mu_0"]
    assert grid["slices"] == law["slices"]


def test_the_plateau_law_jacobian_matches_central_differences() -> None:
    params = np.array([3.0, -1.5, 1.5, 0.5])  # mu_0, kappa_V, u_star, omega_V
    u = np.array([-0.1, 1.0, 1.5, 2.9, 4.9])
    analytic = _plateau_law_jacobian(params, u)
    step = 1e-6
    for column in range(4):
        shift = np.zeros(4)
        shift[column] = step
        numeric = (
            _plateau_law_curve(params + shift, u) - _plateau_law_curve(params - shift, u)
        ) / (2 * step)
        assert analytic[:, column] == pytest.approx(numeric, abs=1e-6)


def test_a_sharp_plateau_corner_on_a_slice_is_held() -> None:
    """The plateaus of the eight length slices of the published measurement (``ū``, ``μ_v`` in
    the coding of the fit, their own SE): the data want a corner sharper than the bound, and it
    settles on the slice 161-320 words. The fit is a cone in ``(u_V, W_V)`` there: an optimizer
    that keeps the parameters strictly inside the range creeps to the bound of the width and does
    not converge. The law must be drawn, with the width held on its lower bound.
    """
    u = np.array([1.950, 2.787, 3.777, 4.703, 5.692, 6.627, 7.541, 8.549])
    c = np.array([14.330, 13.416, 14.385, 12.231, 10.919, 9.091, 7.852, 6.485])
    se = np.array([1.180, 0.708, 0.796, 0.601, 0.542, 0.592, 0.734, 1.051])
    law = _fit_plateau_law(u, c, np.eye(len(u)), np.diag(se**2))
    assert law is not None
    assert law["held"] == ["omega_V"]
    assert law["params"][3] == pytest.approx(0.001, abs=1e-6)
    assert not np.any(law["sensitivity"][3])
