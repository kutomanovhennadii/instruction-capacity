"""The passport of a measurement: from the published points to the passport dictionary.

Steps: the trials of every answered point (the validator); the area of analysis - the purchase
cube and its boundary; the bin layout; point weights by the run cube; for every passport model
- its design and the exclusion of bins without an estimate, its fit, its section and its second
stage - laws, restored bins, public quantities; the collapse of the answer over all answered
points (``collapse``); resources and alarms. The dictionary is the content of
``passport_data.json``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from iec.contracts import MeasurementDescription, PointRecord, PointStatus, Tariff
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.infrastructure.logging import LogContext, get_logger
from iec.passport.analysis_area import build_area_cube, render_cube, safe_points
from iec.passport.approximation import approximation_block, bin_abscissae
from iec.passport.bin_layout import REFERENCE_TOTAL_WORDS, START_D, START_N
from iec.passport.collapse import collapse_block
from iec.passport.design import build_profile_design, drop_degenerate_indicators
from iec.passport.measured_points import MeasuredPoint, measure_points
from iec.passport.model_fitting import HALFWIDTH_QUANTILE, FitResult
from iec.passport.passport_model import PassportModel
from iec.passport.section import expected_sign, vs_grid_passport_section
from iec.passport.vpm_cube import VpmCube, build_layout, cell_balanced_weights
from iec.passport.vspmr_model import VspmrGlmmModel

_MODULE_NAME = "passport_data"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
_SEVERITIES = (SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR)
_SEVERITY_RANK = {SEVERITY_ERROR: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}

#: Alarm code: some billed points carry no provider token counters.
INCOMPLETE_TOKEN_COUNTERS = "INCOMPLETE_TOKEN_COUNTERS"

#: Statuses of a point whose call reached the provider and could be billed.
_BILLED_STATUSES = (PointStatus.OK, PointStatus.API_ERROR)

#: Origin of the covariance: the cluster sandwich by points of the fit of a model.
COVARIANCE_PROVENANCE = "cluster_sandwich"

#: The passport models of the public passport: the passport builds a section for each.
PUBLIC_MODELS: tuple[PassportModel, ...] = (VspmrGlmmModel(),)

_Row = tuple[float, float | None, list[float] | None]


@dataclass(frozen=True, slots=True)
class PassportAlarm:
    """One alarm of the passport - an item of ``alarms`` in ``passport_data.json``.

    ``evidence`` holds the numbers the judgement stands on; ``affects_primary_metrics`` tells
    whether the alarm devalues the primary quantities of the passport.
    """

    severity: str
    code: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    affects_primary_metrics: bool = False

    def __post_init__(self) -> None:
        """Check the severity and the code: measurements are compared by alarm codes."""
        if self.severity not in _SEVERITIES:
            raise ContractViolationError(
                f"alarm severity must be one of {_SEVERITIES}, got '{self.severity}'",
                error_code="PASSPORT_ALARM_INVALID",
            )
        if not self.code or self.code != self.code.upper():
            raise ContractViolationError(
                f"alarm code must be non-empty SCREAMING_SNAKE_CASE, got '{self.code}'",
                error_code="PASSPORT_ALARM_INVALID",
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the alarm as an item of the ``alarms`` list."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "evidence": self.evidence,
            "affects_primary_metrics": self.affects_primary_metrics,
        }


def build_passport_data(
    description: MeasurementDescription,
    records: Sequence[PointRecord],
    name: str,
    models: Sequence[PassportModel] = PUBLIC_MODELS,
) -> dict[str, Any]:
    """Compute the passport of a measurement from its published point records.

    The points, the area of analysis, the layout, the point weights and the bin centres are
    computed once; the design, the fit, the section and the second stage - for every model.

    Args:
        description: Description of the measurement: model, vocabulary, tariff.
        records: Point records in the published order - the order is kept everywhere.
        name: Name of the measurement, the label ``model.experiment_id``.
        models: Passport models, a section for each in this order; the public passport has
            one (``PUBLIC_MODELS``).

    Raises:
        PassportError: ``PASSPORT_INPUT_MISSING`` - there is nothing to fit;
            ``FIT_NOT_CONVERGED`` - a fit did not converge.
        ContractViolationError: ``PASSPORT_MODELS_INVALID`` - no model or a repeated name.
    """
    names = [model.profile_id for model in models]
    if not names or len(set(names)) != len(names):
        raise ContractViolationError(
            f"the passport needs at least one model and distinct names, got {names}",
            error_code="PASSPORT_MODELS_INVALID",
        )
    logger = get_logger(LogContext(module_name=_MODULE_NAME))
    points = measure_points(records, description.vocab_id)
    area = build_area_cube(points)
    safe = safe_points(points, area)
    ok_points = tuple(point for point in safe if point.status is PointStatus.OK)
    if not ok_points:
        raise PassportError(
            "the area of analysis has no ok points: there is nothing to fit",
            error_code="PASSPORT_INPUT_MISSING",
        )
    layout = build_layout(safe)
    weighting = cell_balanced_weights(ok_points, VpmCube(REFERENCE_TOTAL_WORDS))
    run_weights = None if weighting is None else weighting[0]
    cell_weighting = None if weighting is None else weighting[1]
    abscissae = bin_abscissae(ok_points, layout, run_weights, REFERENCE_TOTAL_WORDS)
    fits: dict[str, Any] = {}
    tables: dict[str, Any] = {}
    sections: dict[str, Any] = {}
    for model in models:
        design, dropped = drop_degenerate_indicators(
            build_profile_design(safe, layout, model, START_D, START_N)
        )
        if dropped:
            logger.info(
                f"{model.profile_id}: {len(dropped)} bin(s) without an estimate dropped (all "
                f"replacements right or all wrong): {', '.join(dropped)}"
            )
        fit = model.fit(design, safe, run_weights)
        section = vs_grid_passport_section(
            model,
            fit.theta,
            fit.covariance,
            layout,
            model_id=model.profile_id,
            fit_status={
                "converged": True,
                "regularized": False,
                "separation_detected": fit.separation_detected,
            },
            covariance_provenance=COVARIANCE_PROVENANCE,
        ).as_dict()
        section["approximation"] = approximation_block(
            fit.theta,
            fit.covariance,
            layout,
            abscissae,
            reference_total_words=REFERENCE_TOTAL_WORDS,
            sigma2_in_fit=(
                None
                if fit.sigma2 is None or fit.sigma2_covariance is None
                else (fit.sigma2, fit.sigma2_covariance)
            ),
        )
        rows = _coefficient_rows(fit)
        fits[model.profile_id] = _fit_block(fit, rows, cell_weighting)
        tables[model.profile_id] = _coefficients_table(rows)
        sections[model.profile_id] = section
        logger.info(
            f"passport of '{name}', model {model.profile_id}: {fit.n_runs} point(s) and "
            f"{fit.n_trials} trial(s) in the fit, {len(fit.theta)} parameter(s), "
            f"{len(dropped)} bin(s) excluded"
        )
    resources, alarms = _resources_block(points, description.tariff)
    alarms.sort(key=lambda alarm: (_SEVERITY_RANK[alarm.severity], alarm.code))
    return {
        "model": {"id": description.model.model_id, "experiment_id": name},
        "fit": fits,
        "bernoulli_coefficients": tables,
        "passport": sections,
        "collapse": collapse_block(points),
        "resources": resources,
        "bin_cube": render_cube(area),
        "alarms": [alarm.as_dict() for alarm in alarms],
    }


def _row(value: float, variance: float | None) -> _Row:
    """Return the triple "estimate, SE, 95% interval" of an estimate and its variance."""
    se = math.sqrt(variance) if variance is not None and variance >= 0.0 else None
    ci = None if se is None else [value - HALFWIDTH_QUANTILE * se, value + HALFWIDTH_QUANTILE * se]
    return (value, se, ci)


def _coefficient_rows(fit: FitResult) -> dict[str, _Row]:
    """Return "estimate, SE, 95% interval" by coefficient name, SE from the diagonal."""
    rows: dict[str, _Row] = {}
    for position, (name, value) in enumerate(fit.theta.items()):
        variance = fit.covariance[position][position] if fit.covariance else None
        rows[name] = _row(value, variance)
    return rows


def _coefficients_table(rows: dict[str, _Row]) -> dict[str, Any]:
    """Return the coefficient table: estimate, SE, interval, sign hypothesis, sign mismatch."""
    table: dict[str, Any] = {}
    for name, (value, se, ci) in rows.items():
        sign = expected_sign(name)
        mismatch = sign is not None and (
            (sign == "−" and value > 0.0) or (sign == "+" and value < 0.0)
        )
        table[name] = {
            "estimate": value,
            "se": se,
            "ci95": ci,
            "expected_sign": sign,
            "sign_mismatch": mismatch,
        }
    return table


def _fit_block(
    fit: FitResult, rows: dict[str, _Row], cell_weighting: dict[str, Any] | None
) -> dict[str, Any]:
    """Return the ``fit`` block of a model: θ, SE, intervals, covariance, the variance of the
    point shift with its SE and its origin, sizes, weights.

    ``regularized`` is always ``false``: the passport has no ridge fallback. ``shift_in_fit`` -
    the shift of a point is inside the likelihood of the fit, and the formula of the model
    carries it.
    """
    return {
        "theta": dict(fit.theta),
        "se": {name: se for name, (_, se, _) in rows.items()},
        "ci95": {name: ci for name, (_, _, ci) in rows.items()},
        "covariance": [list(row) for row in fit.covariance],
        "converged": True,
        "regularized": False,
        "separation_detected": fit.separation_detected,
        "sigma2": fit.sigma2,
        "sigma": None if fit.sigma2 is None else math.sqrt(fit.sigma2),
        "sigma2_se": fit.sigma2_se,
        "sigma2_method": fit.sigma2_method,
        "shift_in_fit": fit.sigma2_covariance is not None,
        "rho": fit.rho,
        "n_trials": fit.n_trials,
        "n_runs": fit.n_runs,
        "cell_weighting": cell_weighting,
    }


def _resources_block(
    points: Sequence[MeasuredPoint], tariff: Tariff | None
) -> tuple[dict[str, Any], list[PassportAlarm]]:
    """Return the resources of the published points and their alarm.

    The names are those of the earlier passport. ``attempted_calls`` - points whose call
    reached the provider (``ok`` and ``api_error``); ``unique_point_keys`` - all published
    points; ``ok_runs`` - answered points; ``trials`` - Bernoulli trials of the answered
    points, not model requests; token sums - provider counters where they were reported;
    money - tokens × the tariff of the measurement, the cost of the published points.
    """
    billed = [point for point in points if point.status in _BILLED_STATUSES]
    ok_points = [point for point in points if point.status is PointStatus.OK]
    prompt_tokens = sum(point.record.prompt_tokens or 0 for point in billed)
    response_tokens = sum(point.record.response_tokens or 0 for point in billed)
    without_counters = sum(
        1
        for point in billed
        if point.record.prompt_tokens is None and point.record.response_tokens is None
    )
    input_cost = (
        None if tariff is None else prompt_tokens * tariff.input_usd_per_million / 1e6
    )
    output_cost = (
        None if tariff is None else response_tokens * tariff.output_usd_per_million / 1e6
    )
    total_cost = (
        None if input_cost is None or output_cost is None else input_cost + output_cost
    )
    resources = {
        "attempted_calls": len(billed),
        "unique_point_keys": len(points),
        "ok_runs": len(ok_points),
        "trials": sum(len(point.trials) for point in ok_points),
        "prompt_tokens_billed": prompt_tokens,
        "completion_tokens_billed": response_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }
    alarms: list[PassportAlarm] = []
    if without_counters:
        alarms.append(
            PassportAlarm(
                severity=SEVERITY_WARNING,
                code=INCOMPLETE_TOKEN_COUNTERS,
                message=(
                    "some billed points carry no provider token counters: the token sums "
                    "and the money are understated, and the missing part cannot be recovered"
                ),
                evidence={"records": without_counters},
            )
        )
    return resources, alarms
