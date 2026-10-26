"""Public quantities of the VS-PMR passport - one function per quantity.

IEC by two calculations; the guaranteed instruction capacity IGC - the IEC at full density
p = 1 - by two calculations; fatigue - the price of doubling the position and the fatigue
capacity by the slanted asymptote of the grid law (the word position has no bins of its own:
every cell of the grid belongs to its length slice, so there is no binned calculation);
volume - the price of a doubling; density - the loss at full density; rules - the factor and the scale of the degradation of
late rules, the discount per doubling of the rule set.

A quantity function takes the approximation laws with their sensitivities to the raw
coefficients (the internal form of ``approximation_block``, before the matrices are struck
out) and computes only its own formula; the error is by the delta method over the full
covariance of the fit, extended by ``sigma2`` when the shift of a point is inside the fit: the
threshold of a level depends on it. The name, the unit and the note text do not live here: the
document gives them by key.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iec.passport import second_stage_laws
from iec.passport.bin_layout import BinLayout
from iec.passport.second_stage_laws import (
    STATUS_EXTRAPOLATED,
    STATUS_INSUFFICIENT,
    STATUS_MEASURED,
    STATUS_NOT_DETECTED,
    STATUS_RESERVE_NOT_POSITIVE,
    continuous_iec,
    delta_se,
)
from iec.passport.vspmr_model import (
    _load_axes_price_gradient,
    rule_price,
    vspmr_density_reference,
)

#: Quality levels of the quantities that have a level.
PUBLIC_LEVELS: tuple[float, ...] = (0.95, 0.5)
#: Status: the level is not reached already at the first answer word.
STATUS_BELOW_LEVEL_AT_START = "below_level_at_first_word"

#: Keys of the public quantities in passport order - the document finds the labels by them.
PUBLIC_KEYS: tuple[str, ...] = (
    "iec_capacity",
    "igc_capacity",
    "fatigue_price_per_doubling",
    "fatigue_capacity_words",
    "volume_price_per_doubling",
    "loss_at_full_density",
    "rule_depth_degradation_factor",
    "rule_depth_scale_A_R",
    "rule_set_discount_per_doubling",
)

#: Outcomes of the cumulative IEC. ``defined`` - the reserve is exhausted inside the measured
#: area; ``beyond_rule_price_support`` - it is not exhausted by its bound, a lower bound is
#: known; ``invalid_rule_price_profile`` - the price of a rule is negative in substance.
IEC_STATUS_DEFINED = "defined"
IEC_STATUS_BEYOND_SUPPORT = "beyond_rule_price_support"
IEC_STATUS_INVALID_PROFILE = "invalid_rule_price_profile"

#: Below this value a rule price is negative in substance, not by rounding noise. The scale of
#: the numerical zero of this module; not a tuning knob.
_RULE_PRICE_FLOOR = -1e-9


def rule_price_support_max(layout: BinLayout) -> int:
    """Return the largest rule number whose price has its own coefficient.

    This is the upper bound of the last rule bin: past it there is no free coefficient, and
    there is nothing to continue the sum of prices with. Extrapolating past it is forbidden -
    so the bound is returned explicitly, not implied.
    """
    return layout.row_bounds[-1][1]


@dataclass(frozen=True, slots=True)
class _Crossing:
    """Result of walking the rules: status, whole and fractional capacity, the crossing."""

    status: str
    total: int | None
    fractional: float | None
    crossing_rule: int | None
    cumulative_penalty: float | None


def iec_capacity(target: float, prices: Sequence[float], support_max: int) -> _Crossing:
    """Find the largest ``m`` whose cumulative cost fits into the reserve ``target``.

    The equality contract: ``C(m) ≤ target`` - the table still passes, ``C(m) > target`` -
    it no longer does. The threshold belongs to the passing side.

    The fractional capacity is the same quantity with the unfilled share of the crossing
    rule: ``m_last + (target − C(m_last)) / a(r_crossing)``. The share lies in ``[0, 1)`` by
    construction: ``C(m_last) ≤ target`` gives a non-negative numerator, and
    ``C(m_last) + a(r_crossing) > target`` a numerator below the denominator. The formula does
    not know how the price is set, so it works the same on bins and on a smooth model.

    **A negative price of a single rule does not stop the walk.** The cumulative cost ``C(m)``
    is measured, not each term separately: the estimate of the price of one bin may go below
    zero by fit noise, and the next terms absorb this minus. Stopping the sum at the first
    negative value would refuse the measurement because of noise in one coordinate; a binned
    model can give several negative prices with a positive cumulative cost.

    The profile is declared invalid only when the cumulative cost did not grow on **any**
    rule of the support: then there is nothing to sum, and the capacity is undefined in
    substance, not because of a single outlier.

    The crossing is searched only where the cumulative cost grows: with a non-positive term
    there is nothing to divide the unfilled reserve by, and such a rule cannot be the
    crossing one.

    Returns:
        The result of the walk; with an outcome other than ``defined`` the capacities are
        undefined.
    """
    if target < 0.0:
        return _Crossing(IEC_STATUS_DEFINED, 0, 0.0, None, 0.0)
    cumulative = 0.0
    last_ok = 1
    any_priced = False
    for offset, price in enumerate(prices):
        rule_index = offset + 2
        if not math.isfinite(price):
            return _Crossing(IEC_STATUS_INVALID_PROFILE, None, None, rule_index, None)
        if price >= _RULE_PRICE_FLOOR:
            any_priced = True
            if cumulative + price > target:
                # Crossing only on a growing term: the denominator is strictly positive,
                # and the share lies in [0, 1).
                fractional = last_ok + (target - cumulative) / price
                return _Crossing(
                    IEC_STATUS_DEFINED, last_ok, fractional, rule_index, cumulative + price
                )
        cumulative += price
        last_ok = rule_index
    if not any_priced:
        # No rule of the support had a positive price: the cumulative cost grows nowhere,
        # and the capacity is undefined - a property of the profile, not an outlier.
        return _Crossing(IEC_STATUS_INVALID_PROFILE, None, None, None, cumulative)
    return _Crossing(IEC_STATUS_BEYOND_SUPPORT, None, None, None, cumulative)


@dataclass(frozen=True, slots=True)
class Quantity:
    """A public quantity: key, level, way of calculation, number, error, status.

    ``key`` - the key without the level; ``level`` - the quality level ``q`` or ``None``;
    ``method`` - ``"bins"``, ``"law"`` or ``None`` for quantities with one calculation;
    ``args`` - substitutions into the note of the document (``support`` - the support in the
    units of the quantity).
    """

    key: str
    value: float | None
    se: float | None
    status: str
    level: float | None = None
    method: str | None = None
    args: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the entry for ``passport_data.json``."""
        return {
            "key": self.key,
            "level": self.level,
            "method": self.method,
            "value": self.value,
            "se": self.se,
            "status": self.status,
            "args": dict(self.args),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Quantity:
        """Restore a quantity from an entry of ``passport_data.json``."""
        return cls(
            key=str(row["key"]),
            value=row.get("value"),
            se=row.get("se"),
            status=str(row.get("status")),
            level=row.get("level"),
            method=row.get("method"),
            args=dict(row.get("args") or {}),
        )


@dataclass(frozen=True)
class PublicInputs:
    """Input of the quantity functions: the fit, the laws with sensitivities, the level, shifts.

    ``vs_shift`` - the shift of the grid ``δ_VS`` and its gradient; ``r_shift`` - the shift of the
    ``R`` block and its gradient; ``thresholds`` - the logit of the level ``q`` and its
    gradient.
    """

    theta: Mapping[str, float]
    names: Sequence[str]
    covariance: np.ndarray
    layout: BinLayout
    laws: Mapping[str, Mapping[str, Any]]
    alpha: float
    alpha_gradient: np.ndarray
    vs_shift: tuple[float, np.ndarray]
    r_shift: tuple[float, np.ndarray]
    #: The threshold of every public level: the logit of the median point at which the share
    #: of right replacements over points is ``q``, and its gradient over the extended
    #: coordinates ``(theta, sigma2)`` (:func:`iec.passport.glmm_fit.marginal_threshold`).
    thresholds: Mapping[float, tuple[float, np.ndarray]]


def _support_words(layout: BinLayout) -> int:
    """The largest measured text length - the top of the last length bin."""
    return int(layout.volume_bounds[-1][1])


def _threshold(inputs: PublicInputs, q: float) -> tuple[float, np.ndarray]:
    """Return the threshold of the level ``q`` and its gradient."""
    return inputs.thresholds[q]


# --- IEC ---------------------------------------------------------------------------------


def _binned_iec(
    theta: Mapping[str, float],
    names: Sequence[str],
    covariance: np.ndarray,
    layout: BinLayout,
    reserve: float,
    reserve_gradient: np.ndarray,
    support: int,
    q: float,
    level_key: str,
    price_shift: tuple[float, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return the IEC by bins: the fractional cumulative capacity and its SE by the delta method.

    The walk is :func:`iec_capacity` over the model prices ``a(r) = −ΔM(r) − R(r)``. The
    gradient is the capacity gradient with the reserve given from outside: the model has no
    level ``alpha``.

    ``price_shift`` - the shift of every price and its gradient: the price by the restored bins
    ``θ − s`` equals the raw one plus ``s_R`` (the shift of the ``R`` block); ``None`` - raw
    prices. ``q`` and ``level_key`` are not used - kept for the calls in tests.
    """
    if any(
        f"{prefix}{k}" not in theta for prefix in ("m_bin_", "r_bin_") for k in range(2, layout.R + 1)
    ):
        return {"value": None, "se": None, "status": STATUS_INSUFFICIENT, "args": {"support": support}}
    shift_value, shift_gradient = (0.0, np.zeros(len(names))) if price_shift is None else price_shift
    prices = [rule_price(r, theta, layout) + shift_value for r in range(2, support + 1)]
    crossing = iec_capacity(reserve, prices, support)
    if crossing.status != IEC_STATUS_DEFINED or crossing.fractional is None:
        return {
            "value": float(support) if crossing.status == IEC_STATUS_BEYOND_SUPPORT else None,
            "se": None,
            "status": crossing.status,
            "args": {"support": support},
        }
    se = None
    if crossing.crossing_rule is not None:
        price = prices[crossing.crossing_rule - 2]
        m_last = crossing.total if crossing.total is not None else 1
        margin = reserve - sum(prices[: max(m_last - 1, 0)])
        margin_gradient = reserve_gradient.copy()
        for r in range(2, m_last + 1):
            margin_gradient -= np.array(_load_axes_price_gradient(r, names, layout)) + shift_gradient
        price_gradient = (
            np.array(_load_axes_price_gradient(crossing.crossing_rule, names, layout)) + shift_gradient
        )
        if price > 0.0:
            gradient = margin_gradient / price - margin * price_gradient / (price * price)
            se = delta_se(gradient, covariance)
    return {
        "value": float(crossing.fractional),
        "se": se,
        "status": STATUS_MEASURED,
        "args": {"support": support, "crossing": crossing.crossing_rule, "reserve": reserve},
    }


def iec_binned(inputs: PublicInputs, q: float) -> Quantity:
    """IEC by the restored bins: the cumulative price of rules ``a(r) = −ΔM(r) − R(r)``.

    The reserve is ``α_phys − η*(q, σ²)``; the price of a rule by the bins ``θ − s`` is the raw
    one plus ``s_R``: the capacity is read from the restored bins.
    """
    support = rule_price_support_max(inputs.layout)
    target, target_gradient = _threshold(inputs, q)
    row = _binned_iec(
        inputs.theta,
        inputs.names,
        inputs.covariance,
        inputs.layout,
        inputs.alpha - target,
        inputs.alpha_gradient - target_gradient,
        support,
        q,
        f"{q:g}".replace(".", "_"),
        price_shift=inputs.r_shift,
    )
    return Quantity(
        key="iec_capacity",
        value=row["value"],
        se=row["se"],
        status=row["status"],
        level=q,
        method="bins",
        args=row["args"],
    )


def iec_law(inputs: PublicInputs, q: float) -> Quantity:
    """IEC by the smooth laws: the root of ``C_cont(m) = Δ_q``."""
    support = rule_price_support_max(inputs.layout)
    level_key = f"{q:g}".replace(".", "_")
    target, target_gradient = _threshold(inputs, q)
    row = continuous_iec(
        inputs.covariance,
        inputs.names,
        inputs.alpha - target,
        inputs.alpha_gradient - target_gradient,
        inputs.laws["table_size"],
        inputs.laws["rule_position"],
        support,
        level_key,
        m_smooth=False,
        theta=inputs.theta,
    )[f"iec_continuous_{level_key}"]
    return Quantity(
        key="iec_capacity",
        value=row.get("value"),
        se=row.get("se"),
        status=str(row.get("status")),
        level=q,
        method="law",
        args={"support": support},
    )


# --- Fatigue -----------------------------------------------------------------------------


def _grid(inputs: PublicInputs) -> tuple[Mapping[str, Any], np.ndarray]:
    """Return the grid law in the fit coding and its sensitivity; the rows of the sensitivity are
    ``κ_S, λ_S, μ₀, κ_V, ω_S, u_*, ω_V`` (``approximation.LAW_PARAMETERS``)."""
    law = inputs.laws["vs_grid"]
    return law, np.asarray(law["sensitivity"])


def fatigue_doubling_price(inputs: PublicInputs) -> Quantity:
    """Price of doubling the position: ``−κ_S``, logits per doubling of the word position."""
    law, s = _grid(inputs)
    return Quantity(
        "fatigue_price_per_doubling",
        -float(law["kappa_S"]),
        delta_se(-s[0], inputs.covariance),
        STATUS_MEASURED,
    )


def fatigue_capacity_law(inputs: PublicInputs, q: float) -> Quantity:
    """Fatigue capacity by the law: the inclined asymptote ``κ_S·log₂s + λ_S`` crosses the level.

    The asymptote is one for all text lengths - it is fatigue without the price of length. In
    the restored coding the level is ``α_phys + (κ_S·log₂s + λ_S − δ_VS)``; the root is
    ``log₂s = (η*(q, σ²) − α_phys − (λ_S − δ_VS))/κ_S``.
    """
    law, s = _grid(inputs)
    a, b = float(law["kappa_S"]), float(law["lambda_S"])
    support = _support_words(inputs.layout)
    if a == 0.0:
        return Quantity(
            "fatigue_capacity_words",
            None,
            None,
            STATUS_NOT_DETECTED,
            q,
            "law",
            {"support": support},
        )
    bias, bias_gradient = inputs.vs_shift
    target, target_gradient = _threshold(inputs, q)
    numerator = target - inputs.alpha - (b - bias)
    if numerator >= 0.0:
        return Quantity(
            "fatigue_capacity_words",
            None,
            None,
            STATUS_BELOW_LEVEL_AT_START,
            q,
            "law",
            {"support": support},
        )
    log_x = numerator / a
    words = 2.0**log_x
    gradient = (
        target_gradient - inputs.alpha_gradient - (s[1] - bias_gradient)
    ) / a - numerator / (a * a) * s[0]
    return Quantity(
        "fatigue_capacity_words",
        words,
        delta_se(math.log(2.0) * words * gradient, inputs.covariance),
        STATUS_MEASURED if words <= support else STATUS_EXTRAPOLATED,
        q,
        "law",
        {"support": support},
    )


# --- Volume ------------------------------------------------------------------------------


def volume_doubling_price(inputs: PublicInputs) -> Quantity:
    """Price of doubling the volume: ``−κ_V``, how much the plateau of a long text falls per
    doubling of its length (the slope of the slanted asymptote of the plateau)."""
    law, s = _grid(inputs)
    return Quantity(
        "volume_price_per_doubling",
        -float(law["kappa_V"]),
        delta_se(-s[3], inputs.covariance),
        STATUS_MEASURED,
    )


# --- Density -----------------------------------------------------------------------------


def _full_density_loss(inputs: PublicInputs) -> tuple[float, np.ndarray]:
    """Return the loss at full density by the law ``A_P·(1 − e^(−1/p_c))`` and its gradient."""
    law = inputs.laws["density"]
    s = np.asarray(law["sensitivity"])
    full, (d_a, d_xc) = second_stage_laws.loss_at_full_density(
        float(law["A_P"]), float(law["p_c"])
    )
    return full, d_a * s[1] + d_xc * s[2]


def loss_at_full_density(inputs: PublicInputs) -> Quantity:
    """Loss at full density: ``A_P·(1 − e^(−1/p_c))``."""
    full, gradient = _full_density_loss(inputs)
    return Quantity(
        "loss_at_full_density",
        full,
        delta_se(gradient, inputs.covariance),
        STATUS_MEASURED,
    )


# --- IGC ---------------------------------------------------------------------------------


def _full_density_bin(inputs: PublicInputs) -> tuple[float, np.ndarray] | None:
    """Return the restored coefficient of the density bin ``p = 1`` and its gradient.

    The bin ``p = 1`` is the last density bin of the layout; restored means minus the density
    shift ``μ_P``, as in the restored profile of the passport. The reference bin has the
    implicit raw coefficient zero; a bin removed by the exclusion rule has no estimate -
    ``None``.
    """
    law = inputs.laws["density"]
    shift = float(law["mu_P"])
    gradient = -np.asarray(law["sensitivity"])[0]
    number = inputs.layout.P
    if number == vspmr_density_reference(inputs.layout):
        return -shift, gradient
    name = f"p_bin_{number}"
    if name not in inputs.theta:
        return None
    gradient = gradient.copy()
    gradient[list(inputs.names).index(name)] += 1.0
    return float(inputs.theta[name]) - shift, gradient


def igc_binned(inputs: PublicInputs, q: float) -> Quantity:
    """IGC by the restored bins: the IEC walk with the full-density term in the cumulative price.

    ``C_P(m) = Σ_{r=2..m} a(r) − P(1) ≤ Δ_q``. The term does not depend on ``m``: at ``p = 1``
    every rule brings its own word (``D = m``), so the density stays at one. The walk is the one
    of :func:`iec_binned` with the reserve ``Δ_q + P(1)``, ``P(1)`` the restored bin ``p = 1``;
    a reserve that is not positive means the level is not held even with one rule.
    """
    support = rule_price_support_max(inputs.layout)
    full = _full_density_bin(inputs)
    if full is None:
        return Quantity(
            "igc_capacity", None, None, STATUS_INSUFFICIENT, q, "bins", {"support": support}
        )
    target, target_gradient = _threshold(inputs, q)
    reserve = inputs.alpha - target + full[0]
    if reserve <= 0.0:
        return Quantity(
            "igc_capacity",
            None,
            None,
            STATUS_RESERVE_NOT_POSITIVE,
            q,
            "bins",
            {"support": support},
        )
    row = _binned_iec(
        inputs.theta,
        inputs.names,
        inputs.covariance,
        inputs.layout,
        reserve,
        inputs.alpha_gradient - target_gradient + full[1],
        support,
        q,
        f"{q:g}".replace(".", "_"),
        price_shift=inputs.r_shift,
    )
    return Quantity(
        key="igc_capacity",
        value=row["value"],
        se=row["se"],
        status=row["status"],
        level=q,
        method="bins",
        args=row["args"],
    )


def igc_law(inputs: PublicInputs, q: float) -> Quantity:
    """IGC by the smooth laws: the root of ``C(m) + A_P·(1 − e^(−1/p_c)) = Δ_q``."""
    support = rule_price_support_max(inputs.layout)
    level_key = f"{q:g}".replace(".", "_")
    target, target_gradient = _threshold(inputs, q)
    loss, loss_gradient = _full_density_loss(inputs)
    row = continuous_iec(
        inputs.covariance,
        inputs.names,
        inputs.alpha - target - loss,
        inputs.alpha_gradient - target_gradient - loss_gradient,
        inputs.laws["table_size"],
        inputs.laws["rule_position"],
        support,
        level_key,
        m_smooth=False,
        theta=inputs.theta,
    )[f"iec_continuous_{level_key}"]
    return Quantity(
        key="igc_capacity",
        value=row.get("value"),
        se=row.get("se"),
        status=str(row.get("status")),
        level=q,
        method="law",
        args={"support": support},
    )


# --- Rules -------------------------------------------------------------------------------


def _depth_detected(inputs: PublicInputs) -> bool:
    """Whether the degradation of late rules is detected: ``A_R`` above 1.96·SE."""
    law = inputs.laws["rule_position"]
    s = np.asarray(law["sensitivity"])
    return float(law["A_R"]) > 1.96 * delta_se(s[1], inputs.covariance)


def rule_depth_factor(inputs: PublicInputs) -> Quantity:
    """The degradation factor of late rules ``2^γ``."""
    law = inputs.laws["rule_position"]
    s = np.asarray(law["sensitivity"])
    gamma = float(law["gamma"])
    return Quantity(
        "rule_depth_degradation_factor",
        2.0**gamma,
        delta_se(math.log(2.0) * 2.0**gamma * s[2], inputs.covariance),
        STATUS_MEASURED if _depth_detected(inputs) else STATUS_NOT_DETECTED,
    )


def rule_depth_scale(inputs: PublicInputs) -> Quantity:
    """The degradation scale of late rules ``A_R``."""
    law = inputs.laws["rule_position"]
    s = np.asarray(law["sensitivity"])
    return Quantity(
        "rule_depth_scale_A_R",
        float(law["A_R"]),
        delta_se(s[1], inputs.covariance),
        STATUS_MEASURED if _depth_detected(inputs) else STATUS_NOT_DETECTED,
    )


def rule_set_discount(inputs: PublicInputs) -> Quantity:
    """The discount per doubling of the rule set ``κ_M``."""
    law = inputs.laws["table_size"]
    s = np.asarray(law["sensitivity"])
    kappa_m = float(law["kappa_M"])
    se = delta_se(s[1], inputs.covariance)
    return Quantity(
        "rule_set_discount_per_doubling",
        kappa_m,
        se,
        STATUS_MEASURED if abs(kappa_m) > 1.96 * se else STATUS_NOT_DETECTED,
    )


# --- Assembly ----------------------------------------------------------------------------


def public_quantities(inputs: PublicInputs) -> tuple[Quantity, ...]:
    """Assemble the public quantities in passport order.

    IEC and IGC by two calculations at every level; fatigue - the price of doubling the
    position and the capacity by the asymptote; volume - the price of a doubling; density;
    rules.
    """
    quantities: list[Quantity] = []
    for q in PUBLIC_LEVELS:
        quantities += [iec_binned(inputs, q), iec_law(inputs, q)]
    for q in PUBLIC_LEVELS:
        quantities += [igc_binned(inputs, q), igc_law(inputs, q)]
    quantities.append(fatigue_doubling_price(inputs))
    for q in PUBLIC_LEVELS:
        quantities.append(fatigue_capacity_law(inputs, q))
    quantities.append(volume_doubling_price(inputs))
    quantities.append(loss_at_full_density(inputs))
    quantities += [rule_depth_factor(inputs), rule_depth_scale(inputs), rule_set_discount(inputs)]
    return tuple(quantities)
