"""The paths of the chain: their order, the starts, the turns of the spread by the load."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.telemetry import TelemetryHub
from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.acceleration.sparse_design import build_sparse_design
from iec.passport.fitting.chain.fit_notes import SPREAD_ON_FLOOR, SPREAD_ONE_GROUP
from iec.passport.fitting.chain.fit_paths import (
    FIT_PATHS,
    FIT_PATHS_INVALID,
    METHOD_GLMM,
    METHOD_GLMM_HETEROGENEOUS,
    METHOD_LOGIT,
    METHOD_NONE,
    ChainState,
    FitProblem,
    GlmmPath,
    LoadSpreadPath,
    LogitPath,
    check_chain,
    no_fit,
)
from iec.passport.fitting.glmm_fit import GlmmFitter
from iec.passport.fitting.load_spread import RequestLoad


def _two_spreads() -> np.ndarray:
    """Successes of 160 requests of 40 trials: the spread 0.5 in the first 80, 2.0 in the rest."""
    rng = np.random.default_rng(7)
    shifts = rng.normal(0.0, np.array([0.5] * 80 + [2.0] * 80))
    return rng.binomial(40, 1.0 / (1.0 + np.exp(-(1.0 + shifts)))).astype(float)


def _loads(bins: int) -> dict[str, RequestLoad]:
    """The loads of the 160 requests over ``bins`` length bins, in a row."""
    return {
        f"r{number}": RequestLoad((1 + number * bins // 160, 1, 1), (100.0, 0.25, 2.0))
        for number in range(160)
    }


def _problem(successes: np.ndarray, loads: dict[str, RequestLoad] | None) -> FitProblem:
    """A level over 160 requests of 40 trials with the given successes."""
    rows = FoldedRows(
        x=np.ones((160, 1)),
        successes=successes,
        trials=np.full(160, 40.0),
        run=np.arange(160, dtype=np.int64),
        starts=np.arange(160, dtype=np.int64),
        n_runs=160,
    )
    return FitProblem(
        rows=rows,
        weight=np.ones(160),
        fitter=GlmmFitter(build_sparse_design(rows)),
        run_keys=tuple(f"r{number}" for number in range(160)),
        loads=loads,
    )


def _reached(problem: FitProblem) -> ChainState:
    """Advance the paths of the logit and of the GLMM and return their state."""
    hub = TelemetryHub.log_only()
    logit = LogitPath().advance(problem, ChainState(), hub)
    assert logit.state is not None
    glmm = GlmmPath().advance(problem, logit.state, hub)
    assert glmm.state is not None
    return glmm.state


def test_the_default_chain_goes_from_the_logit_to_the_spread() -> None:
    check_chain(FIT_PATHS)

    assert [path.method for path in FIT_PATHS] == [
        METHOD_LOGIT,
        METHOD_GLMM,
        METHOD_GLMM_HETEROGENEOUS,
    ]
    assert [path.builds_on for path in FIT_PATHS] == [METHOD_NONE, METHOD_LOGIT, METHOD_GLMM]


@pytest.mark.parametrize("starts", [(), (1.0, 0.0)])
def test_the_glmm_path_needs_positive_starts(starts: tuple[float, ...]) -> None:
    with pytest.raises(ConfigError) as caught:
        GlmmPath(starts)
    assert caught.value.error_code == FIT_PATHS_INVALID


def test_without_loads_the_spread_stops_without_a_note() -> None:
    problem = _problem(_two_spreads(), None)

    step = LoadSpreadPath().advance(problem, _reached(problem), TelemetryHub.log_only())

    assert step.state is None and step.notes == ()


def test_a_variance_on_its_floor_stops_the_spread_with_a_note_of_the_level_info() -> None:
    problem = _problem(np.full(160, 29.0), _loads(2))
    state = _reached(problem)
    assert state.glmm is not None and state.glmm.on_floor

    step = LoadSpreadPath().advance(problem, state, TelemetryHub.log_only())

    assert step.state is None
    assert [(note.code, note.level) for note in step.notes] == [(SPREAD_ON_FLOOR, logging.INFO)]


def test_axes_of_one_group_stop_the_spread_with_a_note_of_the_level_info() -> None:
    problem = _problem(_two_spreads(), _loads(1))

    step = LoadSpreadPath().advance(problem, _reached(problem), TelemetryHub.log_only())

    assert step.state is None
    assert [(note.code, note.level) for note in step.notes] == [(SPREAD_ONE_GROUP, logging.INFO)]


def test_the_spread_by_two_groups_gives_the_heterogeneous_estimates() -> None:
    problem = _problem(_two_spreads(), _loads(2))

    step = LoadSpreadPath().advance(problem, _reached(problem), TelemetryHub.log_only())

    assert step.state is not None and step.notes == ()
    estimate = LoadSpreadPath().estimate(problem, step.state)
    assert estimate.method == METHOD_GLMM_HETEROGENEOUS
    assert estimate.spread is not None and estimate.loglik_common is not None
    assert estimate.information.shape == (3, 3)


def test_a_quantity_without_a_fit_has_empty_estimates() -> None:
    estimate = no_fit(7)

    assert estimate.method == METHOD_NONE and estimate.steps == 7
    assert estimate.theta.size == 0 and estimate.covariance.shape == (0, 0)
