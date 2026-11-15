"""The fit of the bins of the three quantities by the passport fit.

The logit of a success by the bins of the axes with the shift of a request inside the
likelihood - the same path as the passport: the plain logit as the start, then the GLMM with
request weights and the cluster sandwich by requests, then the GLMM with heterogeneous
random-intercept variance when the loads of the requests are given. A bin where all trials gave
one outcome is removed with its trials. The sign is that of the passport: a success is the answer
word against its reference word (P_sync), the right replacement (H), no misfire (F); the more
negative a coefficient, the more likely an error.

``VariableFitter`` is the chain of these fits: its paths (``chain.fit_paths``) go in order, and a
path that gives no fit leaves the quantity with the fit reached before it - a shunt, written to
the log with its reason and kept in the notes of the fit. The estimates of the reached path go
through the numerical checks (``chain.numerical_checks``). The folded rows of a quantity and their
sparse design are built once per quantity and handed to all its fits; the quadrature rule is
built once (specification §2.11).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import FIT, TelemetryHub, telemetry_step
from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.acceleration.quadrature_rule import QuadratureRule
from iec.passport.fitting.acceleration.sparse_design import build_sparse_design
from iec.passport.fitting.chain.fit_notes import FIT_NO_DATA, FitNote
from iec.passport.fitting.chain.fit_paths import (
    FIT_PATHS,
    METHOD_NONE,
    ChainState,
    FitPath,
    FitProblem,
    PathEstimate,
    check_chain,
    no_fit,
)
from iec.passport.fitting.chain.numerical_checks import (
    NUMERICAL_CHECKS,
    NumericalCheck,
    check_fit,
)
from iec.passport.fitting.glmm_fit import GLMM_RULE, GlmmFitter
from iec.passport.fitting.load_spread import LoadSpread, RequestLoad
from iec.passport.trials.design.folded_design import VariableDesign

#: The name of the module in the log and in the events of the fits of a quantity.
_MODULE_NAME = "variable_fit"


@dataclass(frozen=True, slots=True)
class VariableFit:
    """The fit of one quantity.

    ``names``, ``theta`` and ``se`` - the remaining columns, their estimates and standard errors
    (the cluster sandwich by requests), ``covariance`` - their covariance; ``dropped`` - columns
    removed by the exclusion rule; ``sigma2`` - the variance of the request shift for method
    ``glmm``; ``sigma2_covariance`` - the covariance row of ``sigma2`` with the coefficients and
    its variance at the end, as ``FitResult.sigma2_covariance`` of the passport (the input of the
    approximation), empty for other methods; ``failures`` and ``trials`` - failures and trials of
    the remaining rows; ``loglik`` - the weighted marginal log-likelihood at the estimates for
    method ``glmm`` (models on the same rows are compared by it), ``None`` for other methods.
    ``spread`` - the law of the spread for method ``glmm_heterogeneous`` (the GLMM with
    heterogeneous random-intercept variance); ``spread_covariance`` - the covariance over
    ``(theta, gamma)``; ``loglik_common`` - the log-likelihood of the fit with one variance on the
    same rows; ``sigma2`` of this method is the variance averaged over the requests with their
    weights. ``notes`` - the notes of the fit in their order: the shunts of its chain and the
    findings of the numerical checks (``chain.fit_notes``).
    """

    variable: str
    method: str
    names: tuple[str, ...]
    theta: tuple[float, ...]
    se: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    dropped: tuple[str, ...]
    sigma2: float | None
    sigma2_covariance: tuple[float, ...]
    steps: int
    rows: int
    points: int
    failures: int
    trials: int
    loglik: float | None = None
    spread: LoadSpread | None = None
    spread_covariance: tuple[tuple[float, ...], ...] = ()
    loglik_common: float | None = None
    notes: tuple[FitNote, ...] = ()

    def coefficient(self, name: str) -> tuple[float, float] | None:
        """Return the estimate and the SE of a column, or ``None`` if the column was removed."""
        if name not in self.names:
            return None
        index = self.names.index(name)
        return self.theta[index], self.se[index]

    def contrast(self, name: str, reference: str) -> tuple[float, float] | None:
        """Return the difference ``θ_name − θ_reference`` and its SE from the fit covariance.

        So an axis that carries the level is drawn: the common uncertainty of the level cancels
        in the difference. ``None`` - one of the columns was removed.
        """
        if name not in self.names or reference not in self.names:
            return None
        i, j = self.names.index(name), self.names.index(reference)
        variance = self.covariance[i][i] + self.covariance[j][j] - 2.0 * self.covariance[i][j]
        return self.theta[i] - self.theta[j], math.sqrt(max(variance, 0.0))


def drop_degenerate_columns(
    x: np.ndarray, successes: np.ndarray, trials: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Apply the exclusion rule of the passport to folded rows.

    A column without remaining rows is removed. If its rows have zero successes or as many
    successes as trials, it has no finite estimate, and the column and its rows are removed.
    Removing rows can make another column degenerate, so the pass repeats to a fixed point.

    Returns:
        The mask of the remaining rows and the numbers of the remaining columns in order.
    """
    keep_rows = np.ones(len(trials), dtype=bool)
    dropped: set[int] = set()
    changed = True
    while changed:
        changed = False
        for column in range(x.shape[1]):
            if column in dropped:
                continue
            active = keep_rows & (x[:, column] != 0.0)
            if not active.any():
                dropped.add(column)
                changed = True
                continue
            hits = float(successes[active].sum())
            total = float(trials[active].sum())
            if hits == 0.0 or hits == total:
                keep_rows &= ~active
                dropped.add(column)
                changed = True
    return keep_rows, [column for column in range(x.shape[1]) if column not in dropped]


class VariableFitter:
    """The chain of fits of a quantity: the paths from the plain logit to the spread by the load.

    The paths go in order (``chain.fit_paths``); each builds on the fit reached before it, and a
    path that gives no fit stops the chain - the quantity keeps the estimates of the last reached
    path. The notes of the paths and the findings of the numerical checks go to the log, one line
    per note, and stay in the result. The quadrature rule is a structure of the setup; the folded
    rows of a quantity and their sparse design are built once per quantity and serve all its fits.

    ``telemetry`` is the hub that records every fit of a quantity and every attempt of a path in
    the block ``fit``.
    """

    telemetry: TelemetryHub

    def __init__(
        self,
        paths: Sequence[FitPath] = FIT_PATHS,
        checks: Sequence[NumericalCheck] = NUMERICAL_CHECKS,
        *,
        quadrature: QuadratureRule = GLMM_RULE,
        telemetry: TelemetryHub | None = None,
    ) -> None:
        """Assemble the chain once.

        Args:
            paths: The paths of the chain in order; by default ``FIT_PATHS``.
            checks: The numerical checks of the reached estimates; by default
                ``NUMERICAL_CHECKS``.
            quadrature: The quadrature rule of the GLMM; by default ``GLMM_RULE``.
            telemetry: Hub that records the fits in the block ``fit``; without it the fitter gets
                a hub of its own that writes the log lines only.

        Raises:
            ConfigError: ``FIT_PATHS_INVALID`` - the chain has no paths, or a path comes before
                the fit it builds on.
        """
        check_chain(paths)
        self._paths = tuple(paths)
        self._checks = tuple(checks)
        self._quadrature = quadrature
        self.telemetry = telemetry if telemetry is not None else TelemetryHub.log_only()
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME, step_name="fit_variable"))

    @telemetry_step("fit_variable", block=FIT, module_name=_MODULE_NAME)
    def fit(
        self,
        design: VariableDesign,
        run_weight: np.ndarray,
        loads: Mapping[str, RequestLoad] | None = None,
    ) -> VariableFit:
        """Fit the logit of a success of a quantity along the chain.

        Args:
            design: The folded matrix of the quantity.
            run_weight: Request weights in the order of ``design.point_keys``.
            loads: The loads of the requests by their keys; ``None`` - the fit with one variance.

        Returns:
            ``VariableFit`` with the method of the last reached path: ``glmm_heterogeneous``,
            ``glmm``, ``logit``, or ``none`` when no path gave a fit (no estimates); its notes
            in ``notes``.

        Side effects:
            One line of the log per note; one event of the block ``fit`` per attempt of a fit
            and one for the whole fit of the quantity.
        """
        successes = design.trials - design.failures
        keep_rows, keep_columns = drop_degenerate_columns(design.x, successes, design.trials)
        kept = set(keep_columns)
        names = tuple(design.names[column] for column in keep_columns)
        dropped = tuple(name for column, name in enumerate(design.names) if column not in kept)
        rows, weight, run_keys = _folded(design, successes, keep_rows, keep_columns, run_weight)
        if rows.n_runs == 0 or not keep_columns:
            message = "no rows or columns remain after the exclusion rule: the quantity has no fit"
            empty_notes = (FitNote(FIT_NO_DATA, message),)
            self._report(design.variable, empty_notes)
            return _result(design.variable, names, dropped, rows, no_fit(0), empty_notes)
        problem = FitProblem(
            rows=rows,
            weight=weight,
            fitter=GlmmFitter(build_sparse_design(rows), self._quadrature),
            run_keys=run_keys,
            loads=loads,
        )
        state = ChainState()
        reached: FitPath | None = None
        spent = 0
        notes: list[FitNote] = []
        for path in self._paths:
            step = path.advance(problem, state, self.telemetry)
            notes.extend(step.notes)
            if step.state is None:
                spent = step.steps
                break
            state, reached = step.state, path
        if reached is None:
            estimate = no_fit(spent)
        else:
            estimate = reached.estimate(problem, state)
            notes.extend(check_fit(estimate, self._checks))
        self._report(design.variable, notes)
        return _result(design.variable, names, dropped, rows, estimate, tuple(notes))

    def _report(self, variable: str, notes: Sequence[FitNote]) -> None:
        """Write one line of the log per note: ``<quantity>: <message> [<code>]``.

        Side effects:
            One record per note with the severity of the note.
        """
        for note in notes:
            self._logger.log(note.level, f"{variable}: {note.message} [{note.code}]")


def fit_variable(
    design: VariableDesign,
    run_weight: np.ndarray,
    loads: Mapping[str, RequestLoad] | None = None,
) -> VariableFit:
    """Fit a quantity once along the default chain.

    A convenience for a single fit: it assembles ``VariableFitter`` with the default paths and
    checks and a hub that writes the log only. A caller that fits several quantities assembles
    one ``VariableFitter`` at its setup and hands it its hub.

    Args:
        design: The folded matrix of the quantity.
        run_weight: Request weights in the order of ``design.point_keys``.
        loads: The loads of the requests by their keys; ``None`` - the fit with one variance.

    Returns:
        ``VariableFit`` - as from ``VariableFitter.fit``.
    """
    return VariableFitter().fit(design, run_weight, loads)


def _folded(
    design: VariableDesign,
    successes: np.ndarray,
    keep_rows: np.ndarray,
    keep_columns: list[int],
    run_weight: np.ndarray,
) -> tuple[FoldedRows, np.ndarray, tuple[str, ...]]:
    """Build ``FoldedRows`` from the remaining rows with request numbers in a row from zero.

    Returns:
        The folded rows, the weight of every remaining request and its key, in the order of the
        rows.
    """
    rows_index = np.flatnonzero(keep_rows)
    old_run = design.run[rows_index]
    unique_runs, new_run = np.unique(old_run, return_inverse=True)
    new_run = new_run.astype(np.int64).ravel()
    starts = (
        np.flatnonzero(np.r_[True, new_run[1:] != new_run[:-1]])
        if len(new_run)
        else np.zeros(0, dtype=np.int64)
    )
    rows = FoldedRows(
        x=np.ascontiguousarray(
            design.x[np.ix_(rows_index, np.asarray(keep_columns, dtype=np.intp))]
        ),
        successes=np.asarray(successes[rows_index], dtype=float),
        trials=np.asarray(design.trials[rows_index], dtype=float),
        run=new_run,
        starts=starts,
        n_runs=int(len(unique_runs)),
    )
    weight = np.asarray(run_weight, dtype=float)[unique_runs]
    run_keys = tuple(design.point_keys[run] for run in unique_runs)
    return rows, weight, run_keys


def _result(
    variable: str,
    names: tuple[str, ...],
    dropped: tuple[str, ...],
    rows: FoldedRows,
    estimate: PathEstimate,
    notes: tuple[FitNote, ...],
) -> VariableFit:
    """Assemble the fit of a quantity from the estimates of the reached path.

    The standard errors come from the diagonal of the covariance, a negative variance taken as
    zero; a quantity without a fit has no names. The counts are those of the remaining rows.
    """
    covariance = estimate.covariance
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0)) if covariance.size else np.zeros(0)
    joint = estimate.spread_covariance
    return VariableFit(
        variable=variable,
        method=estimate.method,
        names=names if estimate.method != METHOD_NONE else (),
        theta=tuple(float(value) for value in estimate.theta),
        se=tuple(float(value) for value in se),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        dropped=dropped,
        sigma2=estimate.sigma2,
        sigma2_covariance=tuple(float(value) for value in estimate.sigma2_row),
        steps=estimate.steps,
        rows=int(len(rows.trials)),
        points=rows.n_runs,
        failures=int((rows.trials - rows.successes).sum()),
        trials=int(rows.trials.sum()),
        loglik=estimate.loglik,
        spread=estimate.spread,
        spread_covariance=(
            () if joint is None else tuple(tuple(float(value) for value in row) for row in joint)
        ),
        loglik_common=estimate.loglik_common,
        notes=notes,
    )
