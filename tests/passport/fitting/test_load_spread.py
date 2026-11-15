"""The spread by the load: the bins of a request, the groups of an axis, the fit by the load."""

from __future__ import annotations

import math

import numpy as np
import pytest

from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin
from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.glmm_fit import fit_glmm, fit_spread
from iec.passport.fitting.load_spread import (
    LOGISTIC_VARIANCE,
    FloorMerge,
    LoadSpread,
    RequestLoad,
    SpreadAxis,
    SpreadDesign,
    fit_by_load,
    fit_spread_merging,
    load_fit,
    load_rho,
    merge_floored_group,
    request_load,
    rho_of_load,
    spread_axes,
    spread_design,
    spread_matrix,
    spread_start,
)
from iec.passport.fitting.logit_fit import fit_failure, fit_logit
from iec.passport.trials.layout.vpm_cube import density_level

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 5120),
    density_bounds=doubling_bounds(1, 256),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 256)),
)
SINGLE = SpreadAxis(groups=((1,),), columns=(None,), centers=(1.0,), shares=(1.0,), failing=(40,))
LENGTH = SpreadAxis(
    groups=((1, 2), (3, 4)),
    columns=(1, None),
    centers=(10.0, 1000.0),
    shares=(0.4, 0.6),
    failing=(40, 40),
)
SPREAD = LoadSpread((0.0, 1.0), (LENGTH, SINGLE, SINGLE))


def _load(v_bin: int, p_bin: int = 1, m_bin: int = 1) -> RequestLoad:
    return RequestLoad((v_bin, p_bin, m_bin), (100.0 * v_bin, 0.25, 2.0))


def test_the_load_of_a_request_reads_the_bins_of_the_layout() -> None:
    load = request_load(LAYOUT, 8, 5, 2)
    assert load.bins == (
        LAYOUT.volume_bin(40),
        max(1, density_level(8, 2) + LAYOUT.P),
        LAYOUT.row_bin(2),
    )
    assert load.values == (40.0, 0.25, 2.0)


def test_rare_edges_merge_and_the_largest_group_is_the_reference() -> None:
    numbers = [1] * 5 + [2] * 40 + [3] * 50 + [4] * 10
    loads = [_load(number) for number in numbers]
    axes = spread_axes(loads, np.ones(len(loads), dtype=bool))
    assert axes[0].groups == ((1, 2), (3, 4))
    assert axes[0].columns == (1, None)
    assert axes[0].failing == (45, 60)
    assert axes[1].columns == (None,)
    matrix = spread_matrix(axes, loads)
    assert matrix.shape == (105, 2)
    assert matrix[0].tolist() == [1.0, 1.0]
    assert matrix[-1].tolist() == [1.0, 0.0]


def test_a_bin_reads_its_group_and_an_unknown_bin_the_nearest() -> None:
    assert SPREAD.row_of_bins((2, 1, 1)).tolist() == [1.0, 1.0]
    assert SPREAD.row_of_bins((4, 1, 1)).tolist() == [1.0, 0.0]
    assert SPREAD.row_of_bins((9, 1, 1)).tolist() == [1.0, 0.0]
    assert SPREAD.row_of_bins((0, 1, 1)).tolist() == [1.0, 1.0]
    assert SPREAD.sigma2_of_bins((2, 1, 1)) == pytest.approx(math.exp(2.0))


@pytest.mark.parametrize(
    ("value", "column"),
    [(10.0, 1.0), (1000.0, 0.0), (100.0, 0.5), (1.0, 1.0), (5000.0, 0.0), (None, 0.4)],
)
def test_a_load_goes_by_the_logarithm_between_the_centres(
    value: float | None, column: float
) -> None:
    assert SPREAD.row_at((value, None, None)) == pytest.approx(np.array([1.0, column]))


def test_the_share_of_the_shift_reads_the_load() -> None:
    assert rho_of_load(0.2, None, _load(1)) == 0.2
    assert rho_of_load(0.2, SPREAD, None) == 0.2
    assert load_rho(0.2, None, LAYOUT, 8, 5, 2) == 0.2
    sigma2 = SPREAD.sigma2_of_bins((1, 1, 1))
    assert rho_of_load(0.2, SPREAD, _load(1)) == pytest.approx(
        sigma2 / (sigma2 + LOGISTIC_VARIANCE)
    )


def _rows_with_two_spreads() -> tuple[FoldedRows, list[RequestLoad]]:
    """Requests of two length bins: the true spread 0.5 in the first and 2.0 in the second."""
    rng = np.random.default_rng(7)
    sigmas = np.array([0.5] * 80 + [2.0] * 80)
    shifts = rng.normal(0.0, sigmas)
    trials = np.full(160, 40.0)
    successes = rng.binomial(40, 1.0 / (1.0 + np.exp(-(1.0 + shifts)))).astype(float)
    rows = FoldedRows(
        x=np.ones((160, 1)),
        successes=successes,
        trials=trials,
        run=np.arange(160, dtype=np.int64),
        starts=np.arange(160, dtype=np.int64),
        n_runs=160,
    )
    return rows, [_load(1)] * 80 + [_load(2)] * 80


def test_the_fit_by_the_load_recovers_two_spreads() -> None:
    rows, loads = _rows_with_two_spreads()
    weight = np.ones(160)
    start = fit_logit(rows, weight)
    common = fit_glmm(rows, weight, start.theta, 1.0)
    found = fit_by_load(rows, weight, common, loads)
    assert found is not None
    assert found.spread.axes[0].columns == (None, 1)
    assert found.spread.gamma[0] == pytest.approx(math.log(0.5), abs=0.3)
    assert found.spread.gamma[1] == pytest.approx(math.log(4.0), abs=0.45)
    assert found.fit.loglik > common.loglik
    assert 1.2 < found.sigma2 < 3.4
    assert found.sigma2_row.shape == (2,)


def test_axes_of_one_group_leave_the_one_variance() -> None:
    rows, _ = _rows_with_two_spreads()
    weight = np.ones(160)
    start = fit_logit(rows, weight)
    common = fit_glmm(rows, weight, start.theta, 1.0)
    assert fit_by_load(rows, weight, common, [_load(1)] * 160) is None


def test_the_steps_of_the_fit_by_the_load_give_the_fit_by_the_load() -> None:
    rows, loads = _rows_with_two_spreads()
    weight = np.ones(160)
    common = fit_glmm(rows, weight, fit_logit(rows, weight).theta, 1.0)
    design = spread_design(rows, loads)
    start = spread_start(common, design)

    found = fit_spread(rows, weight, common.theta, start, design.matrix, common.modes)

    assert design.grouped
    assert fit_failure(found.converged, found.steps, found.theta, found.gamma) is None
    by_steps = load_fit(found, design, weight)
    whole = fit_by_load(rows, weight, common, loads)
    assert whole is not None
    assert by_steps.spread == whole.spread
    assert by_steps.sigma2 == whole.sigma2
    assert np.array_equal(by_steps.sigma2_row, whole.sigma2_row)


def test_a_design_of_one_group_on_every_axis_is_not_grouped() -> None:
    rows, _ = _rows_with_two_spreads()
    design = spread_design(rows, [_load(1)] * 160)
    assert not design.grouped
    assert design.matrix.shape == (160, 1)


def _four_groups() -> tuple[SpreadDesign, list[RequestLoad], np.ndarray]:
    """Four length bins, every request failing: 40, 60, 50 and 45 requests; bin 2 is the reference."""
    loads = [_load(1)] * 40 + [_load(2)] * 60 + [_load(3)] * 50 + [_load(4)] * 45
    failed = np.ones(len(loads), dtype=bool)
    axes = spread_axes(loads, failed)
    return SpreadDesign(axes, spread_matrix(axes, loads)), loads, failed


def test_a_group_at_the_edge_merges_with_its_only_neighbour() -> None:
    design, loads, failed = _four_groups()
    assert design.axes[0].columns == (1, None, 2, 3)

    merged = merge_floored_group(design, np.array([0.0, -9.0, 0.1, 0.2]), loads, failed)

    assert merged is not None
    new_design, merge = merged
    assert merge == FloorMerge(axis=0, bins=(1,), into=(2,))
    assert new_design.axes[0].groups == ((1, 2), (3,), (4,))
    assert new_design.axes[0].columns == (None, 1, 2)
    assert new_design.matrix.shape == (195, 3)


def test_a_group_in_the_middle_merges_with_the_neighbour_of_fewer_failures() -> None:
    design, loads, failed = _four_groups()

    merged = merge_floored_group(design, np.array([0.0, 0.3, -9.0, 0.2]), loads, failed)

    assert merged is not None
    assert merged[1] == FloorMerge(axis=0, bins=(3,), into=(4,))
    assert merged[0].axes[0].groups == ((1,), (2,), (3, 4))


def test_the_reference_group_with_the_lowest_spread_merges_too() -> None:
    """The reference has no column: its parameter is zero, and its free neighbours climb above it."""
    design, loads, failed = _four_groups()

    merged = merge_floored_group(design, np.array([-9.0, 9.0, 9.5, 9.7]), loads, failed)

    assert merged is not None
    assert merged[1] == FloorMerge(axis=0, bins=(2,), into=(1,))


def test_axes_of_one_group_have_nothing_to_merge() -> None:
    loads = [_load(1)] * 50
    failed = np.ones(50, dtype=bool)
    axes = spread_axes(loads, failed)
    design = SpreadDesign(axes, spread_matrix(axes, loads))
    assert merge_floored_group(design, np.array([0.0]), loads, failed) is None


def _rows_with_a_group_without_spread(seed: int) -> tuple[FoldedRows, list[RequestLoad]]:
    """Three length bins of 80 requests: no spread in the first, 1 and 2 in the others."""
    rng = np.random.default_rng(seed)
    sigmas = np.array([0.0] * 80 + [1.0] * 80 + [2.0] * 80)
    shifts = rng.normal(0.0, 1.0, 240) * sigmas
    successes = rng.binomial(40, 1.0 / (1.0 + np.exp(-(1.0 + shifts)))).astype(float)
    rows = FoldedRows(
        x=np.ones((240, 1)),
        successes=successes,
        trials=np.full(240, 40.0),
        run=np.arange(240, dtype=np.int64),
        starts=np.arange(240, dtype=np.int64),
        n_runs=240,
    )
    return rows, [_load(1)] * 80 + [_load(2)] * 80 + [_load(3)] * 80


def test_a_fit_stopped_on_the_floor_merges_the_group_and_converges() -> None:
    rows, loads = _rows_with_a_group_without_spread(7)
    weight = np.ones(240)
    common = fit_glmm(rows, weight, fit_logit(rows, weight).theta, 1.0)
    design = spread_design(rows, loads)
    stuck = fit_spread(rows, weight, common.theta, spread_start(common, design), design.matrix)

    found = fit_by_load(rows, weight, common, loads)

    assert stuck.floored and not stuck.converged
    assert found is not None
    assert found.merges == (FloorMerge(axis=0, bins=(1,), into=(2,)),)
    assert found.spread.axes[0].groups == ((1, 2), (3,))
    assert found.fit.converged and not found.fit.floored


def test_a_converged_fit_merges_nothing() -> None:
    rows, loads = _rows_with_two_spreads()
    weight = np.ones(160)
    common = fit_glmm(rows, weight, fit_logit(rows, weight).theta, 1.0)

    found = fit_by_load(rows, weight, common, loads)

    assert found is not None and found.merges == ()


class _Stuck:
    """A fit of the spread that stops on the floor while the first length bin has its own group."""

    def __init__(self, design: SpreadDesign) -> None:
        self.floored = design.axes[0].groups[0] == (1,)
        self.gamma = np.array([0.0, -9.0, 0.1, 0.2][: design.matrix.shape[1]])


def test_the_merging_repeats_the_fit_until_it_leaves_the_floor() -> None:
    design, loads, failed = _four_groups()
    seen: list[SpreadDesign] = []

    def fit_once(current: SpreadDesign) -> _Stuck:
        seen.append(current)
        return _Stuck(current)

    last, found, merges = fit_spread_merging(design, fit_once, loads, failed)

    assert len(seen) == 2
    assert found is not None and not found.floored
    assert last.axes[0].groups == ((1, 2), (3,), (4,))
    assert merges == (FloorMerge(axis=0, bins=(1,), into=(2,)),)


def test_merges_down_to_one_group_leave_no_fit_of_the_spread() -> None:
    loads = [_load(1)] * 40 + [_load(2)] * 60
    failed = np.ones(100, dtype=bool)
    axes = spread_axes(loads, failed)
    design = SpreadDesign(axes, spread_matrix(axes, loads))

    last, found, merges = fit_spread_merging(design, _Stuck, loads, failed)

    assert found is None
    assert not last.grouped
    assert merges == (FloorMerge(axis=0, bins=(1,), into=(2,)),)
