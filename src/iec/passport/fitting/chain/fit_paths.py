"""The paths of the chain of fits of a quantity: each builds a richer model on the fit before it.

The chain goes from the plain logit to the GLMM with one variance of the request shift and on to
the GLMM with heterogeneous random-intercept variance, where the variance depends on the load. A
path that gives no fit stops the chain, and the quantity keeps the fit reached before it: this
fallback is a shunt (specification §12.8), and the path tells why in a note. Every path is a
strategy of one interface (§2.10): it declares the method it gives and the method it builds on,
``check_chain`` rejects an order where a path comes before the fit it needs, and the default chain
is ``FIT_PATHS``. Every attempt of a fit is an event of the block ``fit`` of the telemetry.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, TypeVar

import numpy as np

from iec.infrastructure.exceptions import ConfigError, ContractViolationError
from iec.infrastructure.telemetry import FIT, TelemetryHub
from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.chain.fit_notes import (
    GLMM_FAILED,
    GLMM_START_FAILED,
    LOGIT_FAILED,
    SPREAD_FAILED,
    SPREAD_MERGED,
    SPREAD_ON_FLOOR,
    SPREAD_ONE_GROUP,
    FitNote,
)
from iec.passport.fitting.glmm_fit import GlmmFit, GlmmFitter, SpreadFit
from iec.passport.fitting.load_spread import (
    LoadFit,
    LoadSpread,
    RequestLoad,
    SpreadDesign,
    failing_requests,
    fit_spread_merging,
    load_fit,
    spread_design,
    spread_start,
)
from iec.passport.fitting.logit_fit import LogitFit, fit_failure, fit_logit

#: The name of the module in the events of the attempts.
_MODULE_NAME = "fit_paths"
#: Error code of a chain whose paths come before the fits they build on, or of bad starts.
FIT_PATHS_INVALID = "FIT_PATHS_INVALID"
#: The starts of the variance of the request shift, tried in order until a fit converges; the
#: coefficients start from the plain logit. A full Newton step from a variance far below the maximum
#: can overshoot it many times over, and the fit then crawls along a flat slope: on a measurement of
#: 3564 points the start 1 reached the variance 129 and did not converge in 100 steps, while the
#: starts 4, 5, 8, 16 and 64 reached the same maximum, the variance 13.93, in 26-28 steps.
SIGMA2_STARTS: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
#: Fit method: the shift of a request inside the likelihood.
METHOD_GLMM = "glmm"
#: Fit method: the GLMM with heterogeneous random-intercept variance - the shift of a request
#: whose variance depends on the load.
METHOD_GLMM_HETEROGENEOUS = "glmm_heterogeneous"
#: Fit method: the logit without the shift - the shift did not converge.
METHOD_LOGIT = "logit"
#: The fit did not converge at all.
METHOD_NONE = "none"

_FitT = TypeVar("_FitT")


@dataclass(frozen=True, slots=True, eq=False)
class FitProblem:
    """What every path of a quantity fits.

    ``rows`` - the folded rows of the quantity; ``weight`` - the weight of every request;
    ``fitter`` - the GLMM fitter of these rows, assembled once; ``run_keys`` - the key of every
    request in the order of the rows; ``loads`` - the loads of the requests by their keys,
    ``None`` - the fit with one variance.
    """

    rows: FoldedRows
    weight: np.ndarray
    fitter: GlmmFitter
    run_keys: tuple[str, ...]
    loads: Mapping[str, RequestLoad] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class ChainState:
    """The fits the chain has reached: the plain logit, the GLMM and the fit by the load."""

    logit: LogitFit | None = None
    glmm: GlmmFit | None = None
    by_load: LoadFit | None = None


@dataclass(frozen=True, slots=True, eq=False)
class PathStep:
    """What a path gives.

    ``state`` - the state with the fit of the path, ``None`` - the path gives no fit and the chain
    stops; ``notes`` - the notes of the path; ``steps`` - the Newton steps the path spent without
    a fit, which become the steps of a quantity that has no fit at all.
    """

    state: ChainState | None
    notes: tuple[FitNote, ...] = ()
    steps: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class PathEstimate:
    """The estimates of the reached path in the form of the fit of a quantity.

    ``theta`` and ``covariance`` - the coefficients and their covariance; ``sigma2`` and
    ``sigma2_row`` - the variance of the request shift and its covariance row with the
    coefficients and its own variance at the end, ``None`` and empty without the shift; ``steps``
    - the Newton steps of the path; ``information`` - the information whose pseudo-inverse is
    the bread of the sandwich, for the numerical checks; ``loglik``, ``spread``,
    ``spread_covariance`` and ``loglik_common`` - as in the fit of a quantity.
    """

    method: str
    theta: np.ndarray
    covariance: np.ndarray
    sigma2: float | None
    sigma2_row: np.ndarray
    steps: int
    information: np.ndarray
    loglik: float | None = None
    spread: LoadSpread | None = None
    spread_covariance: np.ndarray | None = None
    loglik_common: float | None = None


class FitPath(Protocol):
    """One path of the chain of fits of a quantity."""

    @property
    def method(self) -> str:
        """Return the method of the fit the path gives."""
        ...

    @property
    def builds_on(self) -> str:
        """Return the method of the fit the path needs before it."""
        ...

    def advance(self, problem: FitProblem, state: ChainState, telemetry: TelemetryHub) -> PathStep:
        """Try the fit of the path on the fits reached before it.

        Args:
            problem: What the quantity fits.
            state: The fits the chain has reached.
            telemetry: The hub that records every attempt of a fit in the block ``fit``.

        Returns:
            The state with the fit of the path, or a step without a state and with the reason.
        """
        ...

    def estimate(self, problem: FitProblem, state: ChainState) -> PathEstimate:
        """Return the estimates of the path from the state that holds its fit."""
        ...


def _require(fit: _FitT | None, method: str) -> _FitT:
    """Return the fit a path builds on.

    Raises:
        ContractViolationError: ``FIT_PATHS_INVALID`` - the chain has not reached the method: its
            paths were assembled out of order without ``check_chain``.
    """
    if fit is None:
        raise ContractViolationError(
            f"a path needs the fit of the method {method}, which the chain has not reached",
            error_code=FIT_PATHS_INVALID,
        )
    return fit


class LogitPath:
    """The plain logit: the start of every chain; without it the quantity has no fit."""

    method = METHOD_LOGIT
    builds_on = METHOD_NONE

    def advance(self, problem: FitProblem, state: ChainState, telemetry: TelemetryHub) -> PathStep:
        """Fit the plain logit by Newton steps; a fit that is not usable stops the chain."""
        with telemetry.observed(FIT, "fit_logit", module_name=_MODULE_NAME):
            start = fit_logit(problem.rows, problem.weight)
        failure = fit_failure(start.converged, start.steps, start.theta)
        if failure is not None:
            note = FitNote(LOGIT_FAILED, f"the plain logit {failure}: the quantity has no fit")
            return PathStep(None, (note,), start.steps)
        return PathStep(replace(state, logit=start))

    def estimate(self, problem: FitProblem, state: ChainState) -> PathEstimate:
        """Return the coefficients of the plain logit and their cluster sandwich."""
        start = _require(state.logit, METHOD_LOGIT)
        return PathEstimate(
            method=METHOD_LOGIT,
            theta=start.theta,
            covariance=np.asarray(start.covariance),
            sigma2=None,
            sigma2_row=np.zeros(0),
            steps=start.steps,
            information=start.information,
        )


@dataclass(frozen=True, slots=True)
class GlmmPath:
    """The GLMM with one variance of the request shift, from the starts of the variance in order.

    A start that gives no fit is a note of the level ``INFO``, and the next start follows. When
    no start gives a fit, the quantity keeps the plain logit.
    """

    method = METHOD_GLMM
    builds_on = METHOD_LOGIT
    starts: tuple[float, ...] = SIGMA2_STARTS

    def __post_init__(self) -> None:
        """Check the starts: at least one, every one positive.

        Raises:
            ConfigError: ``FIT_PATHS_INVALID`` - no starts, or a start that is not positive.
        """
        if not self.starts or min(self.starts) <= 0.0:
            raise ConfigError(
                f"the GLMM needs positive starts of the variance, got {self.starts}",
                error_code=FIT_PATHS_INVALID,
            )

    def advance(self, problem: FitProblem, state: ChainState, telemetry: TelemetryHub) -> PathStep:
        """Fit the GLMM from every start in order until a fit is usable."""
        logit = _require(state.logit, METHOD_LOGIT)
        notes: list[FitNote] = []
        for sigma2_start in self.starts:
            with telemetry.observed(FIT, "fit_glmm", module_name=_MODULE_NAME):
                glmm = problem.fitter.fit(problem.weight, logit.theta, sigma2_start)
            failure = fit_failure(glmm.converged, glmm.steps, glmm.theta)
            if failure is None:
                return PathStep(replace(state, glmm=glmm), tuple(notes))
            message = f"the GLMM from the start sigma2 = {sigma2_start:g} {failure}"
            notes.append(FitNote(GLMM_START_FAILED, message, logging.INFO))
        starts = ", ".join(f"{value:g}" for value in self.starts)
        message = (
            f"the GLMM gave no fit from the starts {starts}: the quantity keeps the plain logit"
        )
        notes.append(FitNote(GLMM_FAILED, message))
        return PathStep(None, tuple(notes))

    def estimate(self, problem: FitProblem, state: ChainState) -> PathEstimate:
        """Return the coefficients and the variance of the GLMM and their cluster sandwich."""
        glmm = _require(state.glmm, METHOD_GLMM)
        size = problem.rows.x.shape[1]
        full = np.asarray(glmm.covariance)
        free = size if glmm.on_floor else size + 1
        return PathEstimate(
            method=METHOD_GLMM,
            theta=glmm.theta,
            covariance=full[:size, :size],
            sigma2=float(glmm.sigma2),
            sigma2_row=full[size, : size + 1],
            steps=glmm.steps,
            information=glmm.information[:free, :free],
            loglik=float(glmm.loglik),
        )


class LoadSpreadPath:
    """The GLMM with heterogeneous random-intercept variance: the variance depends on the load.

    Without loads the quantity asked for one variance, and the path stops without a note. A
    variance on its floor, axes of one group and a group merged because the fit stopped on the
    floor of the spread are turns that the rule foresees (notes of the level ``INFO``); a fit that
    is not usable is a fallback to one variance.
    """

    method = METHOD_GLMM_HETEROGENEOUS
    builds_on = METHOD_GLMM

    def advance(self, problem: FitProblem, state: ChainState, telemetry: TelemetryHub) -> PathStep:
        """Fit the spread by the load from the GLMM with one variance."""
        glmm = _require(state.glmm, METHOD_GLMM)
        by_key = problem.loads
        if by_key is None:
            return PathStep(None)
        if glmm.on_floor:
            message = (
                "the variance of the request shift stays on its floor: "
                "the spread by the load is not fitted"
            )
            return PathStep(None, (FitNote(SPREAD_ON_FLOOR, message, logging.INFO),))
        loads = [by_key[key] for key in problem.run_keys]
        design = spread_design(problem.rows, loads)
        if not design.grouped:
            message = "every axis of the spread has one group: the quantity keeps one variance"
            return PathStep(None, (FitNote(SPREAD_ONE_GROUP, message, logging.INFO),))

        def fit_once(current: SpreadDesign) -> SpreadFit:
            """Fit the spread of a design from the GLMM with one variance."""
            with telemetry.observed(FIT, "fit_spread", module_name=_MODULE_NAME):
                return problem.fitter.fit_spread(
                    problem.weight,
                    glmm.theta,
                    spread_start(glmm, current),
                    current.matrix,
                    glmm.modes,
                )

        design, found, merges = fit_spread_merging(
            design, fit_once, loads, failing_requests(problem.rows)
        )
        notes = tuple(FitNote(SPREAD_MERGED, merge.text, logging.INFO) for merge in merges)
        if found is None:
            message = "every axis of the spread has one group: the quantity keeps one variance"
            return PathStep(None, (*notes, FitNote(SPREAD_ONE_GROUP, message, logging.INFO)))
        failure = fit_failure(found.converged, found.steps, found.theta, found.gamma)
        if failure is not None:
            message = (
                f"the fit with the spread by the load {failure}: the quantity keeps one variance"
            )
            return PathStep(None, (*notes, FitNote(SPREAD_FAILED, message)))
        by_load = load_fit(found, design, problem.weight, merges)
        return PathStep(replace(state, by_load=by_load), notes)

    def estimate(self, problem: FitProblem, state: ChainState) -> PathEstimate:
        """Return the estimates of the fit by the load and its joint covariance."""
        by_load = _require(state.by_load, METHOD_GLMM_HETEROGENEOUS)
        glmm = _require(state.glmm, METHOD_GLMM)
        size = problem.rows.x.shape[1]
        joint = by_load.fit.covariance
        return PathEstimate(
            method=METHOD_GLMM_HETEROGENEOUS,
            theta=by_load.fit.theta,
            covariance=joint[:size, :size],
            sigma2=by_load.sigma2,
            sigma2_row=by_load.sigma2_row,
            steps=by_load.fit.steps,
            information=by_load.fit.information,
            loglik=float(by_load.fit.loglik),
            spread=by_load.spread,
            spread_covariance=joint,
            loglik_common=float(glmm.loglik),
        )


#: The default chain: the plain logit, the GLMM with one variance, the spread by the load.
FIT_PATHS: tuple[FitPath, ...] = (LogitPath(), GlmmPath(), LoadSpreadPath())


def check_chain(paths: Sequence[FitPath]) -> None:
    """Check that every path comes right after the fit it builds on.

    Raises:
        ConfigError: ``FIT_PATHS_INVALID`` - the chain has no paths, or a path builds on a method
            that the chain does not reach right before it.
    """
    if not paths:
        raise ConfigError("the chain of fits has no paths", error_code=FIT_PATHS_INVALID)
    reached = METHOD_NONE
    for path in paths:
        if path.builds_on != reached:
            raise ConfigError(
                f"the path {path.method} builds on {path.builds_on}, "
                f"but the chain reaches {reached} before it",
                error_code=FIT_PATHS_INVALID,
            )
        reached = path.method


def no_fit(steps: int) -> PathEstimate:
    """Return the estimates of a quantity without a fit: method ``none`` and empty arrays.

    Args:
        steps: The Newton steps spent without a fit.
    """
    empty = np.zeros(0)
    return PathEstimate(
        method=METHOD_NONE,
        theta=empty,
        covariance=np.zeros((0, 0)),
        sigma2=None,
        sigma2_row=empty,
        steps=steps,
        information=np.zeros((0, 0)),
    )
