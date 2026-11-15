"""The fit of a quantity: known coefficients, the starts, the shunts of the chain, their notes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.telemetry import FIT, MemorySink, ObservationLevels, TelemetryHub
from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.chain import fit_paths
from iec.passport.fitting.chain.fit_notes import (
    FIT_NO_DATA,
    GLMM_FAILED,
    GLMM_START_FAILED,
    LOGIT_FAILED,
)
from iec.passport.fitting.chain.fit_paths import (
    FIT_PATHS_INVALID,
    METHOD_GLMM,
    METHOD_GLMM_HETEROGENEOUS,
    METHOD_LOGIT,
    METHOD_NONE,
    SIGMA2_STARTS,
    FitPath,
    GlmmPath,
    LoadSpreadPath,
    LogitPath,
)
from iec.passport.fitting.glmm_fit import GlmmFit, GlmmFitter
from iec.passport.fitting.load_spread import RequestLoad
from iec.passport.fitting.logit_fit import LogitFit
from iec.passport.fitting.variable_fit import VariableFitter, drop_degenerate_columns, fit_variable
from iec.passport.trials.design.folded_design import VariableDesign


def _synthetic_design(seed: int) -> tuple[VariableDesign, dict[str, float]]:
    """400 requests, two grid cells carry the level, bins M 2 and 3 relative to the reference M 1.

    ``truth`` - the logit of a success (the passport sign); the failure probability -
    ``1/(1 + e^η)``.
    """
    truth = {"vs_1_1": 3.0, "vs_2_1": 2.0, "m_bin_2": -0.5, "m_bin_3": -1.0}
    rng = np.random.default_rng(seed)
    rows, failures, trials, run = [], [], [], []
    for point in range(400):
        shift = rng.normal(0.0, 0.5)
        cell = [1.0, 0.0] if point % 2 == 0 else [0.0, 1.0]
        for m_columns in ([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]):
            x = np.array(cell + m_columns)
            eta = float(x @ np.array(list(truth.values()))) + shift
            rows.append(x)
            trials.append(200.0)
            failures.append(float(rng.binomial(200, 1.0 / (1.0 + np.exp(eta)))))
            run.append(point)
    design = VariableDesign(
        variable="H",
        x=np.array(rows),
        failures=np.array(failures),
        trials=np.array(trials),
        run=np.array(run, dtype=np.int64),
        point_keys=tuple(str(point) for point in range(400)),
        names=tuple(truth),
        cells=(),
        axes={},
        abscissae={},
    )
    return design, truth


def test_variable_fit_recovers_known_coefficients() -> None:
    design, truth = _synthetic_design(seed=20261031)

    fit = fit_variable(design, np.ones(400))

    assert fit.method == METHOD_GLMM
    assert fit.sigma2 is not None and 0.1 < fit.sigma2 < 0.5
    for name, value in truth.items():
        estimate = fit.coefficient(name)
        assert estimate is not None
        assert abs(estimate[0] - value) < 3.0 * estimate[1], name
    contrast = fit.contrast("vs_2_1", "vs_1_1")
    assert contrast is not None
    assert abs(contrast[0] + 1.0) < 3.0 * contrast[1]
    assert fit.failures == int(design.failures.sum())
    assert fit.contrast("vs_2_1", "vs_9_9") is None


def _failing_below(starts: list[float], converging_from: float) -> Callable[..., GlmmFit]:
    """Return a stand-in of ``GlmmFitter.fit`` that keeps its starts.

    Below ``converging_from`` the stand-in gives its fit unconverged.
    """
    fit = GlmmFitter.fit

    def failing(
        fitter: GlmmFitter, weight: np.ndarray, theta: np.ndarray, sigma2_start: float
    ) -> GlmmFit:
        starts.append(sigma2_start)
        fitted = fit(fitter, weight, theta, sigma2_start)
        return fitted if sigma2_start >= converging_from else replace(fitted, converged=False)

    return failing


def test_fit_starts_again_when_the_first_start_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, _ = _synthetic_design(seed=20261031)
    reference = fit_variable(design, np.ones(400))
    starts: list[float] = []
    monkeypatch.setattr(GlmmFitter, "fit", _failing_below(starts, SIGMA2_STARTS[1]))

    fit = fit_variable(design, np.ones(400))

    assert starts == list(SIGMA2_STARTS[:2])
    assert fit.method == METHOD_GLMM
    assert fit.sigma2 == pytest.approx(reference.sigma2, rel=1e-6)
    assert fit.theta == pytest.approx(reference.theta, rel=1e-6)


def test_fit_gives_the_logit_when_no_start_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    design, _ = _synthetic_design(seed=20261031)
    starts: list[float] = []
    monkeypatch.setattr(GlmmFitter, "fit", _failing_below(starts, float("inf")))

    fit = fit_variable(design, np.ones(400))

    assert starts == list(SIGMA2_STARTS)
    assert fit.method == METHOD_LOGIT
    assert fit.sigma2 is None


def test_degenerate_column_is_removed_with_its_rows() -> None:
    x = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    successes = np.array([3.0, 0.0, 0.0])
    trials = np.array([10.0, 10.0, 10.0])

    keep_rows, keep_columns = drop_degenerate_columns(x, successes, trials)

    # Column 2 has no rows; column 1 has no successes - it goes with rows 1 and 2.
    assert keep_columns == [0]
    assert keep_rows.tolist() == [True, False, False]


def test_with_loads_the_fit_goes_by_the_load_and_without_them_by_one_variance() -> None:
    rng = np.random.default_rng(11)
    sigmas = np.array([0.5] * 80 + [2.0] * 80)
    shifts = rng.normal(0.0, sigmas)
    successes = rng.binomial(40, 1.0 / (1.0 + np.exp(-(1.0 + shifts))))
    keys = tuple(f"r{number}" for number in range(160))
    design = VariableDesign(
        variable="H",
        x=np.ones((160, 1)),
        failures=(40 - successes).astype(float),
        trials=np.full(160, 40.0),
        run=np.arange(160, dtype=np.int64),
        point_keys=keys,
        names=("level",),
        cells=(),
        axes={},
        abscissae={},
    )
    loads = {
        key: RequestLoad((1 if number < 80 else 2, 1, 1), (100.0, 0.25, 2.0))
        for number, key in enumerate(keys)
    }
    plain = fit_variable(design, np.ones(160))
    assert plain.method == METHOD_GLMM
    assert plain.spread is None
    fit = fit_variable(design, np.ones(160), loads)
    assert fit.method == METHOD_GLMM_HETEROGENEOUS
    assert fit.spread is not None
    assert fit.loglik > plain.loglik
    assert fit.loglik_common == pytest.approx(plain.loglik)
    assert len(fit.sigma2_covariance) == 2
    assert np.asarray(fit.spread_covariance).shape == (3, 3)


def _two_spreads() -> tuple[VariableDesign, dict[str, RequestLoad]]:
    """160 requests of one column: the spread 0.5 in the first length bin and 2.0 in the second."""
    rng = np.random.default_rng(11)
    sigmas = np.array([0.5] * 80 + [2.0] * 80)
    shifts = rng.normal(0.0, sigmas)
    successes = rng.binomial(40, 1.0 / (1.0 + np.exp(-(1.0 + shifts))))
    keys = tuple(f"r{number}" for number in range(160))
    design = VariableDesign(
        variable="H",
        x=np.ones((160, 1)),
        failures=(40 - successes).astype(float),
        trials=np.full(160, 40.0),
        run=np.arange(160, dtype=np.int64),
        point_keys=keys,
        names=("level",),
        cells=(),
        axes={},
        abscissae={},
    )
    loads = {
        key: RequestLoad((1 if number < 80 else 2, 1, 1), (100.0, 0.25, 2.0))
        for number, key in enumerate(keys)
    }
    return design, loads


def test_a_start_without_a_fit_is_a_note_of_the_level_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, _ = _synthetic_design(seed=20261031)
    monkeypatch.setattr(GlmmFitter, "fit", _failing_below([], SIGMA2_STARTS[1]))

    fit = fit_variable(design, np.ones(400))

    assert fit.method == METHOD_GLMM
    assert [(note.code, note.level) for note in fit.notes] == [(GLMM_START_FAILED, logging.INFO)]
    assert "the GLMM from the start sigma2 = 1 did not converge" in fit.notes[0].message


def test_no_start_with_a_fit_leaves_the_logit_and_a_warning_in_the_log(
    monkeypatch: pytest.MonkeyPatch,
    isolated_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The package logger does not propagate records, so stderr is read, as in the other tests
    # of the log; the fixture gives the logger without handlers, bound to the capsys stream.
    design, _ = _synthetic_design(seed=20261031)
    monkeypatch.setattr(GlmmFitter, "fit", _failing_below([], float("inf")))

    fit = fit_variable(design, np.ones(400))

    starts = len(SIGMA2_STARTS)
    assert [note.code for note in fit.notes] == [GLMM_START_FAILED] * starts + [GLMM_FAILED]
    assert [note.level for note in fit.notes] == [logging.INFO] * starts + [logging.WARNING]
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "H: the GLMM gave no fit from the starts 1, 4, 16, 64" in err
    assert "[GLMM_FAILED]" in err


def test_a_logit_without_a_fit_leaves_the_quantity_without_a_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design, _ = _synthetic_design(seed=20261031)
    plain = fit_paths.fit_logit
    spent: list[int] = []

    def failing(rows: FoldedRows, weight: np.ndarray) -> LogitFit:
        fitted = plain(rows, weight)
        spent.append(fitted.steps)
        return replace(fitted, converged=False)

    monkeypatch.setattr(fit_paths, "fit_logit", failing)

    fit = fit_variable(design, np.ones(400))

    assert fit.method == METHOD_NONE
    assert fit.names == () and fit.theta == () and fit.sigma2 is None
    assert fit.steps == spent[0]
    assert [note.code for note in fit.notes] == [LOGIT_FAILED]


def test_a_quantity_without_columns_has_no_fit_and_a_note() -> None:
    design = VariableDesign(
        variable="F",
        x=np.ones((4, 1)),
        failures=np.zeros(4),
        trials=np.full(4, 10.0),
        run=np.arange(4, dtype=np.int64),
        point_keys=("a", "b", "c", "d"),
        names=("level",),
        cells=(),
        axes={},
        abscissae={},
    )

    fit = fit_variable(design, np.ones(4))

    assert fit.method == METHOD_NONE and fit.steps == 0
    assert fit.dropped == ("level",) and fit.points == 0
    assert [note.code for note in fit.notes] == [FIT_NO_DATA]


def test_every_fit_of_a_quantity_is_an_event_of_the_block_fit() -> None:
    design, loads = _two_spreads()
    sink = MemorySink()
    fitter = VariableFitter(telemetry=TelemetryHub(ObservationLevels(), [sink]))

    fit = fitter.fit(design, np.ones(160), loads)

    assert fit.method == METHOD_GLMM_HETEROGENEOUS
    glmm_attempts = 1 + sum(note.code == GLMM_START_FAILED for note in fit.notes)
    operations = [event.operation for event in sink.events]
    assert operations == (
        ["fit_logit"] + ["fit_glmm"] * glmm_attempts + ["fit_spread", "fit_variable"]
    )
    assert {event.block for event in sink.events} == {FIT}


@pytest.mark.parametrize(
    "paths",
    [(), (GlmmPath(), LogitPath()), (LogitPath(), LoadSpreadPath())],
)
def test_a_chain_out_of_order_is_a_configuration_error(paths: tuple[FitPath, ...]) -> None:
    with pytest.raises(ConfigError) as caught:
        VariableFitter(paths)
    assert caught.value.error_code == FIT_PATHS_INVALID


def test_a_chain_without_the_spread_fits_one_variance_with_loads() -> None:
    design, loads = _two_spreads()

    fit = VariableFitter((LogitPath(), GlmmPath())).fit(design, np.ones(160), loads)

    assert fit.method == METHOD_GLMM and fit.spread is None
