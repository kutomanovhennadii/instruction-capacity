"""Second stage of the passport: laws through the fitted bins, and the continuous IEC.

The second stage is **not** a Bernoulli fit but a measuring treatment of the already
estimated bins: a law is drawn through the bin coefficients of a block by generalised least
squares over the full covariance of the coefficients - the line ``λ_M + κ_M·log₂ m`` for the
number of rules, the curve ``λ_R − A_R·(r^γ − 1)`` for the rule number, the saturating curve
``μ_P − A_P·(1 − e^(−p/p_c))`` for the density. The value of a law at the zero of its axis is the
shift of the block; moving it into the level is a calibration, not a change of the model: the
prediction of every trial stays the same.

Uncertainties are by the delta method through the first-stage covariance: every second-stage
estimate is a function of the bin coefficients, and its sensitivity to them is known -
explicitly for a linear fit, through the Gauss-Newton normal equations for a non-linear one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from iec.passport.bin_layout import BinLayout
from iec.passport.logit_fit import INFORMATION_RCOND

#: Statuses of published quantities. ``measured`` - the quantity is computed; the others say
#: honestly why there is no number or what limits it.
STATUS_MEASURED = "measured"
STATUS_INSUFFICIENT = "insufficient_support"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_EXTRAPOLATED = "extrapolated"
STATUS_BEYOND_SUPPORT = "beyond_rule_price_support"
STATUS_RESERVE_NOT_POSITIVE = "reserve_not_positive"
#: The phenomenon is not found in the window of the measurement: the law preferred the form
#: without it.
STATUS_NOT_DETECTED = "not_detected"

#: Minimum points of the second stage: the number of law parameters plus one. The ``M`` line
#: carries two parameters, the ``R`` curve three. Fewer - the support is not covered by the
#: measurement, and the numbers are not published.
_MIN_POINTS_M = 3
_MIN_POINTS_R = 4


# --- Points of the second stage ---------------------------------------------------------


def axis_points(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    prefix: str,
    bounds: Sequence[tuple[int, int]],
    abscissae: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return the points of an axis for the second stage: abscissae, values, Σ block, indices.

    The abscissa of a bin is its **top**: the grid steps are built by the same doubling rule as
    the bin bounds, and the ordered point stands exactly at the top. The reference bin is
    skipped: its coefficient is an implicit zero, not a measured point, and including it would
    substitute the coding for the measurement.

    ``abscissae`` - abscissae given from outside by bin name (the centre of mass of the trials,
    when the nodes do not stand at the bin tops); a bin without an entry gets its top. The
    points are ordered by abscissa - without outside abscissae this is the order of numbers.
    """
    order = list(names)
    selected = sorted(
        (
            float(abscissae[f"{prefix}{index}"])
            if abscissae is not None and f"{prefix}{index}" in abscissae
            else float(bounds[index - 1][1]),
            f"{prefix}{index}",
        )
        for index in range(2, len(bounds) + 1)
        if f"{prefix}{index}" in order
    )
    x = np.array([abscissa for abscissa, _ in selected], dtype=float)
    beta = np.array([float(theta[name]) for _, name in selected], dtype=float)
    idx = [order.index(name) for _, name in selected]
    block = covariance[np.ix_(idx, idx)] if idx else np.zeros((0, 0))
    return x, beta, block, idx


def _precision(block: np.ndarray) -> np.ndarray:
    """Return the weight matrix of generalised least squares - the pseudo-inverse Σ block.

    The bin coefficients come from one fit and are correlated, so the weights are taken from
    the full block, not from the diagonal: ordinary least squares would understate the error.
    Pseudo-inversion with the threshold ``INFORMATION_RCOND`` - a direction below the threshold
    is not supported by the data.
    """
    return np.linalg.pinv(block, rcond=INFORMATION_RCOND, hermitian=True)


def _sensitivity(
    jacobian: np.ndarray, precision: np.ndarray, idx: Sequence[int], width: int
) -> np.ndarray:
    """Return the sensitivity of the law estimates to the raw coefficients: ``(JᵀPJ)⁻¹JᵀP``.

    Second-stage estimates are functions of the bin coefficients, and at the solution their
    derivative with respect to these coefficients equals this matrix (the Gauss-Newton normal
    equations). Through it the delta method carries the first-stage covariance to every
    passport quantity: the second stage has no separate source of uncertainty.
    """
    normal = jacobian.T @ precision @ jacobian
    rows = np.linalg.solve(normal, jacobian.T @ precision)
    full = np.zeros((rows.shape[0], width), dtype=float)
    full[:, list(idx)] = rows
    return full


def fit_m_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    row_bounds: Sequence[tuple[int, int]],
    abscissae: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit the line ``M(m) = λ_M + κ_M·log₂ m`` through the table-size bins.

    ``κ_M`` is the change of the level per doubling of the table, ``λ_M`` the line at ``m = 1``.
    Generalised least squares over the full Σ block; the abscissae are the bin tops (or
    ``abscissae`` by name), the reference bin is not a point. Returns the estimates and the
    sensitivity matrix to the raw ``θ``: the delta method builds the errors of every dependent
    quantity from it.
    """
    x, beta, block, idx = axis_points(theta, covariance, names, "m_bin_", row_bounds, abscissae)
    if x.size < _MIN_POINTS_M:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    design = np.column_stack([np.ones_like(x), np.log2(x)])
    precision = _precision(block)
    coefficients = np.linalg.solve(design.T @ precision @ design, design.T @ precision @ beta)
    sensitivity = _sensitivity(design, precision, idx, len(names))
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        "lambda_M": float(coefficients[0]),
        "kappa_M": float(coefficients[1]),
        "sensitivity": sensitivity,
        "support_max": int(x[-1]),
    }


def _r_law_residual(
    parameters: np.ndarray, x: np.ndarray, beta: np.ndarray, root: np.ndarray
) -> np.ndarray:
    """Return the whitened residual of the ``R`` curve: ``Lᵀ(β − f(x))``."""
    level, amplitude, gamma = parameters
    model = level - amplitude * (np.power(x, gamma) - 1.0)
    return root.T @ (beta - model)


def _r_law_jacobian(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``∂f/∂(λ_R, A_R, γ)`` of the late-rule curve."""
    _, amplitude, gamma = parameters
    power = np.power(x, gamma)
    return np.column_stack([np.ones_like(x), -(power - 1.0), -amplitude * power * np.log(x)])


def fit_r_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    row_bounds: Sequence[tuple[int, int]],
    abscissae: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit the curve ``R(r) = λ_R − A_R·(r^γ − 1)`` through the rule-position bins.

    ``λ_R`` is the value of the curve at the first rule ``r = 1``. Non-linear generalised least
    squares: the residuals are whitened by the root of the weight matrix (``P = L Lᵀ`` from the
    eigendecomposition), then plain ``least_squares``. The zero ``R(1) = 0`` is not imposed in
    advance: that would lose the calibration shift itself, which the second stage has to
    measure.

    Start: ``λ_R`` - the first measured coefficient, the scale ``A_R`` - the slope between the
    ends, the exponent ``γ`` - one. Bounds: ``A_R ≥ 0``, ``γ ≥ 0`` - otherwise the "degradation"
    of late rules would change sign and stop being a degradation. ``abscissae`` - bin abscissae
    from outside, as in :func:`axis_points`.
    """
    x, beta, block, idx = axis_points(theta, covariance, names, "r_bin_", row_bounds, abscissae)
    if x.size < _MIN_POINTS_R:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    precision = _precision(block)
    eigenvalues, vectors = np.linalg.eigh(precision)
    keep = eigenvalues > 0.0
    root = vectors[:, keep] * np.sqrt(eigenvalues[keep])
    span = float(x[-1] - x[0])
    slope = float(beta[0] - beta[-1]) / span if span else 1e-3
    start = np.array([float(beta[0]), max(slope, 1e-3), 1.0], dtype=float)
    result = least_squares(
        _r_law_residual,
        start,
        args=(x, beta, root),
        bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
    )
    if not result.success:
        return {
            "status": STATUS_NOT_CONVERGED,
            "n_points": int(x.size),
            "optimizer_message": str(result.message),
        }
    level, amplitude, gamma = (float(value) for value in result.x)
    sensitivity = _sensitivity(_r_law_jacobian(result.x, x), precision, idx, len(names))
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        "lambda_R": level,
        "A_R": amplitude,
        "gamma": gamma,
        "sensitivity": sensitivity,
        "support_max": int(x[-1]),
    }


# --- Density law ------------------------------------------------------------------------

#: Minimum points of the density curve - the number of its parameters plus one, the same rule
#: as for ``M`` and ``R``: the curve carries three parameters.
_MIN_POINTS_P = 4


def _named_points(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    prefix: str,
    abscissa: Any,
    abscissae: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    """Return the points of a series by name: abscissae, values, Σ block, indices, names.

    **Names** are walked, not numbers: the reference density bin is not the first one, and its
    name is absent from the fit. A bin without trials (zero variance - an empty design column)
    is not a point: its coefficient is not measured. A bin with separation has a huge error,
    and generalised least squares gives it a negligible weight by itself - no separate rule is
    needed.

    ``abscissae`` - abscissae from outside by name; a name without an entry gets
    ``abscissa(number)``. Sorting by abscissa is after the substitution, otherwise the points
    of the law and the rows of the sensitivity would part by index.
    """
    order = list(names)

    def place(name: str) -> float:
        if abscissae is not None and name in abscissae:
            return float(abscissae[name])
        return float(abscissa(int(name[len(prefix) :])))

    selected = sorted(
        (place(name), name)
        for name in order
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
    )
    selected = [
        (x, name) for x, name in selected if covariance[order.index(name), order.index(name)] > 0.0
    ]
    x = np.array([value for value, _ in selected], dtype=float)
    beta = np.array([float(theta[name]) for _, name in selected], dtype=float)
    idx = [order.index(name) for _, name in selected]
    block = covariance[np.ix_(idx, idx)] if idx else np.zeros((0, 0))
    return x, beta, block, idx, [name for _, name in selected]


def _whitening(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the weight matrix of generalised least squares and its root ``L`` (``P = L Lᵀ``)."""
    precision = _precision(block)
    eigenvalues, vectors = np.linalg.eigh(precision)
    keep = eigenvalues > 0.0
    return precision, vectors[:, keep] * np.sqrt(eigenvalues[keep])


def _gls_least_squares(
    curve: Any,
    jacobian: Any,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    root: np.ndarray,
    method: str = "trf",
) -> tuple[Any, float]:
    """Solve non-linear GLS on the whitened residuals ``Lᵀ(β − f(x))``; return ``χ²`` too.

    ``method`` - ``"trf"`` by default; ``"dogbox"`` for a law whose parameter may be held on a
    bound of its range: it treats a bound the data push into as active and stops on it, while
    ``"trf"`` keeps the parameters strictly inside and approaches the bound only asymptotically.
    """
    result = least_squares(
        lambda params: root.T @ (beta - curve(params, x)),
        np.clip(np.asarray(start, dtype=float), lower, upper),
        jac=lambda params: -root.T @ jacobian(params, x),
        bounds=(lower, upper),
        method=method,
    )
    return result, float(result.fun @ result.fun)


def _points_payload(names: Sequence[str], x: np.ndarray) -> list[dict[str, Any]]:
    """Return the points of a series for the passport: the bin name and its abscissa."""
    return [{"name": name, "x": float(value)} for name, value in zip(names, x, strict=True)]


def density_abscissa(layout: BinLayout, index: int) -> float:
    """Return the abscissa of a density bin in ``p = m/D``: the geometric centre of the bin.

    A density bin is an interval ``[lo, hi]`` of the ratio ``D // m``; ``p ≈ 1/(D // m)``, and
    the centre of the bin in ``p`` is ``1/√(lo·hi)``: for geometrically widening bins the
    coordinate is taken at the geometric centre.
    """
    lower, upper = layout.density_bounds[index - 1]
    return 1.0 / math.sqrt(float(lower) * float(upper))


def _p_curve(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The second-stage density curve: ``μ_P − A_P·(1 − e^(−x/p_c))``."""
    bias, amplitude, scale = params
    return bias - amplitude * (1.0 - np.exp(-x / scale))


def _p_jacobian(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return ``∂f/∂(μ_P, A_P, p_c)`` of the density curve."""
    _, amplitude, scale = params
    decay = np.exp(-x / scale)
    return np.column_stack([np.ones_like(x), -(1.0 - decay), amplitude * decay * x / scale**2])


def fit_p_law(
    theta: Mapping[str, float],
    covariance: np.ndarray,
    names: Sequence[str],
    layout: BinLayout,
    abscissae: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit the density curve ``P(p) = μ_P − A_P·(1 − e^(−p/p_c))`` through the bins.

    Non-linear generalised least squares over the full Σ block; the abscissa of a bin is
    :func:`density_abscissa` or ``abscissae`` by name. The shift ``μ_P`` is the value of the
    curve at ``p → 0`` in the fit coding; the physical law is ``−A_P·(1 − e^(−p/p_c))``. Two
    starts (from the data and a fixed one), the better by ``χ²`` is taken.
    """
    x, beta, block, idx, point_names = _named_points(
        theta,
        covariance,
        names,
        "p_bin_",
        lambda index: density_abscissa(layout, index),
        abscissae,
    )
    if x.size < _MIN_POINTS_P:
        return {"status": STATUS_INSUFFICIENT, "n_points": int(x.size)}
    precision, root = _whitening(block)
    lower = np.array([-np.inf, 1e-6, 1e-6])
    upper = np.array([np.inf, np.inf, np.inf])
    low_density = float(beta[int(np.argmin(x))])
    starts = (
        np.array([low_density, max(low_density - float(beta.min()), 0.1), float(np.median(x))]),
        np.array([low_density, 1.0, 0.5]),
    )
    result, chi2 = min(
        (
            _gls_least_squares(_p_curve, _p_jacobian, start, lower, upper, x, beta, root)
            for start in starts
        ),
        key=lambda pair: pair[1],
    )
    try:
        sensitivity = _sensitivity(_p_jacobian(result.x, x), precision, idx, len(names))
    except np.linalg.LinAlgError:
        sensitivity = None
    if not result.success or sensitivity is None:
        return {
            "status": STATUS_NOT_CONVERGED,
            "n_points": int(x.size),
            "optimizer_message": str(result.message),
        }
    bias, amplitude, scale = (float(value) for value in result.x)
    return {
        "status": STATUS_MEASURED,
        "n_points": int(x.size),
        "mu_P": bias,
        "A_P": amplitude,
        "p_c": scale,
        "chi2": chi2,
        "points": _points_payload(point_names, x),
        "sensitivity": sensitivity,
    }


# --- Derived quantities -----------------------------------------------------------------


def _metric(
    value: float | None,
    gradient: np.ndarray | None,
    covariance: np.ndarray,
    unit: str,
    note: str,
    *,
    status: str = STATUS_MEASURED,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble a published row: the value, SE by the delta method, the unit, a note."""
    se = None
    if gradient is not None and value is not None:
        se = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
    return {"status": status, "value": value, "se": se, "unit": unit, "note": note, **extra}


def _law_gradient(names: Sequence[str], parts: Mapping[str, float]) -> np.ndarray:
    """Return the gradient over the raw ``θ`` of a quantity that depends only on law parameters.

    The law parameters are the fit coordinates themselves, so the gradient is a sparse vector
    of their derivatives: the second stage takes no part here.
    """
    order = list(names)
    gradient = np.zeros(len(order))
    for name, derivative in parts.items():
        gradient[order.index(name)] = derivative
    return gradient


# --- Continuous IEC ---------------------------------------------------------------------


def _cumulative_cost(m: float, kappa_m: float, amplitude: float, gamma: float) -> float:
    """Return ``C_cont(m)`` - the cumulative price of a rule set as an integral.

    C_cont(m) = −κ_M·log₂ m + A_R·((m^(γ+1) − 1)/(γ+1) − (m − 1))
    """
    return -kappa_m * math.log2(m) + amplitude * (
        (m ** (gamma + 1.0) - 1.0) / (gamma + 1.0) - (m - 1.0)
    )


def _cumulative_derivatives(
    m: float, kappa_m: float, amplitude: float, gamma: float
) -> tuple[float, float, float, float]:
    """Return ``(∂C/∂m, ∂C/∂κ_M, ∂C/∂A_R, ∂C/∂γ)`` at ``m``."""
    power = m ** (gamma + 1.0)
    d_m = -kappa_m / (m * math.log(2.0)) + amplitude * (m**gamma - 1.0)
    d_kappa = -math.log2(m)
    d_amplitude = (power - 1.0) / (gamma + 1.0) - (m - 1.0)
    d_gamma = amplitude * (
        (power * math.log(m)) / (gamma + 1.0) - (power - 1.0) / (gamma + 1.0) ** 2
    )
    return d_m, d_kappa, d_amplitude, d_gamma


def continuous_iec(
    covariance: np.ndarray,
    names: Sequence[str],
    reserve: float,
    reserve_gradient: np.ndarray,
    m_fit: Mapping[str, Any],
    r_fit: Mapping[str, Any],
    support_max: int,
    level_key: str,
    *,
    m_smooth: bool,
    theta: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Return the continuous IEC of a level: the root of ``C_cont(m) = Δ_q`` and its uncertainty.

    A second calculation next to the binned one: the binned one answers "what is measured",
    the integral one "what the adopted physical form gives", and a discrepancy between them is
    read as the quality of the approximation, not as an error.

    The root is searched by Brent's method inside the support of the rule prices; past the
    support no number is published - the instrument does not extrapolate there, and a lower
    bound is more honest. The uncertainty is by the delta method through the implicit
    function: ``dm/dp = −(∂(C−R)/∂p)/(∂C/∂m)``.
    """
    key = f"iec_continuous_{level_key}"
    if r_fit.get("status") != STATUS_MEASURED:
        return {
            key: _metric(
                None,
                None,
                covariance,
                "rules",
                "the late-rule curve is not measured by the second stage",
                status=str(r_fit.get("status", STATUS_INSUFFICIENT)),
            )
        }
    amplitude = float(r_fit["A_R"])
    gamma = float(r_fit["gamma"])
    if m_smooth:
        # A model with a smooth M carries its slope per unit of ln m as the coefficient
        # ``beta_M``; the cumulative price takes the slope per doubling.
        kappa_m = math.log(2.0) * float(theta["beta_M"])
        kappa_gradient = math.log(2.0) * _law_gradient(names, {"beta_M": 1.0})
    elif m_fit.get("status") == STATUS_MEASURED:
        kappa_m = float(m_fit["kappa_M"])
        kappa_gradient = m_fit["sensitivity"][1]
    else:
        return {
            key: _metric(
                None,
                None,
                covariance,
                "rules",
                "the table-size line is not measured by the second stage",
                status=str(m_fit.get("status", STATUS_INSUFFICIENT)),
            )
        }

    def excess(m: float) -> float:
        return _cumulative_cost(m, kappa_m, amplitude, gamma) - reserve

    if reserve <= 0.0:
        return {
            key: _metric(
                None,
                None,
                covariance,
                "rules",
                "the level reserve is exhausted: the level is already below the target quality",
                status=STATUS_RESERVE_NOT_POSITIVE,
            )
        }
    top = float(support_max)
    if excess(top) < 0.0:
        return {
            key: _metric(
                None,
                None,
                covariance,
                "rules",
                f"the cumulative price does not exhaust the reserve over the whole support (up "
                f"to {support_max} rules): a lower bound is published",
                status=STATUS_BEYOND_SUPPORT,
                lower_bound=support_max,
            )
        }
    root = float(brentq(excess, 1.0, top, xtol=1e-10))
    d_m, d_kappa, d_amplitude, d_gamma = _cumulative_derivatives(root, kappa_m, amplitude, gamma)
    if abs(d_m) < 1e-12:
        return {
            key: _metric(
                root,
                None,
                covariance,
                "rules",
                "the cumulative price is flat at the crossing: no interval is built",
            )
        }
    # dm/dθ = −[∂C/∂θ − ∇Δ_q] / (∂C/∂m); the derivatives of the second-stage parameters are
    # carried to the raw coordinates by their sensitivity.
    gradient = (
        -(
            d_kappa * kappa_gradient
            + d_amplitude * r_fit["sensitivity"][1]
            + d_gamma * r_fit["sensitivity"][2]
            - reserve_gradient
        )
        / d_m
    )
    return {
        key: _metric(
            root,
            gradient,
            covariance,
            "rules",
            f"continuous calculation by the smooth law; support up to {support_max} rules",
            support_max_rules=support_max,
        )
    }


# --- Helpers of the delta method and the density loss -----------------------------------


def delta_se(gradient: np.ndarray, covariance: np.ndarray) -> float:
    """Return the SE by the delta method: ``√(∇′·Cov·∇)``; a noisy negative square is zero."""
    return math.sqrt(max(float(gradient @ covariance @ gradient), 0.0))


def increment_matrix(order: Sequence[str | None]) -> np.ndarray:
    """Return the matrix ``A`` of the move to neighbouring increments ``Δ = A·β`` in physical order.

    ``None`` in the order is a reference position with an implicit zero: the increment through
    it is taken from zero. Comparing blocks by increments does not depend on which level is
    chosen as the reference: changing the reference bin shifts all coefficients by a constant,
    and the increments do not see it.

    Returns:
        A ``(L − 1) × k`` matrix, where ``L`` is the length of the order and ``k`` the number of
        names; the columns are in the order of appearance of the names. For the order
        "neg, None, pos" - ``[[−1, 0], [0, 1]]``.
    """
    named = [name for name in order if name is not None]
    column = {name: position for position, name in enumerate(named)}
    matrix = np.zeros((len(order) - 1, len(named)), dtype=float)
    for row in range(1, len(order)):
        right, left = order[row], order[row - 1]
        if right is not None:
            matrix[row - 1, column[right]] += 1.0
        if left is not None:
            matrix[row - 1, column[left]] -= 1.0
    return matrix


def loss_at_full_density(
    A_P: float,  # noqa: N803
    p_c: float,
) -> tuple[float, tuple[float, float]]:
    """Return the loss when every word is replaced ``A_P·(1 − e^(−1/p_c))`` and its gradient.

    The gradient is over ``(A_P, p_c)``: ``1 − e^(−1/p_c)`` and ``−A_P·e^(−1/p_c)/p_c²``.
    """
    expo = math.exp(-1.0 / p_c)
    return A_P * (1.0 - expo), (1.0 - expo, -A_P * expo / (p_c * p_c))
