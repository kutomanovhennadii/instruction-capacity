"""The passport fit: the GLMM fit, no fallback on non-convergence."""

import dataclasses
import math

import numpy as np
import pytest

from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport import model_fitting
from iec.passport.design import Design
from iec.passport.logit_fit import FoldedRows, LogitFit
from iec.passport.model_fitting import fit_vspmr_glmm


def _design(outcomes: tuple[int, ...]) -> Design:
    rows = [[1.0, 0.0]] * 8 + [[0.0, 1.0]] * 8
    keys = ("a",) * 4 + ("b",) * 4 + ("c",) * 4 + ("d",) * 4
    return Design(matrix=np.array(rows), outcomes=outcomes, run_keys=keys, names=("x", "y"))


_OUTCOMES = (1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0)


def test_without_a_spread_the_fit_recovers_the_logits_of_two_cells() -> None:
    """Two indicator columns, four points without a spread: 6 of 8 right, then 2 of 8.

    The points of a cell agree, so the variance of the shift stays on its floor and the
    coefficients are the logits of the cells.
    """
    fit = fit_vspmr_glmm(_design(_OUTCOMES), None)
    assert fit.theta["x"] == pytest.approx(math.log(3.0), abs=1e-6)
    assert fit.theta["y"] == pytest.approx(math.log(1.0 / 3.0), abs=1e-6)
    assert (fit.n_trials, fit.n_runs) == (16, 4)
    assert fit.sigma2 == pytest.approx(1e-8)
    assert fit.sigma2_se is None
    assert fit.rho == pytest.approx(0.0, abs=1e-8)
    assert fit.sigma2_covariance is not None
    assert len(fit.covariance) == 2 and len(fit.sigma2_covariance) == 3


def test_a_glmm_fit_that_does_not_converge_stops_the_passport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = model_fitting.fit_glmm
    monkeypatch.setattr(
        model_fitting,
        "fit_glmm",
        lambda *args: dataclasses.replace(real(*args), converged=False),
    )
    with pytest.raises(PassportError) as error:
        fit_vspmr_glmm(_design(_OUTCOMES), None)
    assert error.value.error_code == "FIT_NOT_CONVERGED"


def test_a_fit_that_does_not_converge_stops_the_passport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stalled(rows: FoldedRows, run_weight: np.ndarray, start: object = None) -> LogitFit:
        size = rows.x.shape[1]
        return LogitFit(
            theta=np.zeros(size),
            information=np.eye(size),
            covariance=np.eye(size),
            converged=False,
            separated=False,
            steps=55,
        )

    monkeypatch.setattr(model_fitting, "fit_logit", stalled)
    with pytest.raises(PassportError) as error:
        fit_vspmr_glmm(_design(_OUTCOMES), None)
    assert error.value.error_code == "FIT_NOT_CONVERGED"


def test_an_empty_design_is_nothing_to_fit() -> None:
    design = Design(matrix=np.zeros((0, 2)), outcomes=(), run_keys=(), names=("x", "y"))
    with pytest.raises(PassportError) as error:
        fit_vspmr_glmm(design, None)
    assert error.value.error_code == "PASSPORT_INPUT_MISSING"


def test_a_point_of_the_design_without_a_weight_is_refused() -> None:
    with pytest.raises(ContractViolationError) as error:
        fit_vspmr_glmm(_design(_OUTCOMES), {"a": 1.0})
    assert error.value.error_code == "FIT_RUN_WEIGHTS_MISSING"
