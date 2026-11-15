"""The law of a block: its established form unless an alternative is much better, without outliers.

A block of a quantity names its forms in the assignment of forms (``config/passport_laws.yaml``):
the established form first, then the alternatives - the line, the hyperbola, the slanted hyperbola
and the exponential. The law goes through every bin of the block with a coefficient, and the
outliers leave it by the clipping of astropy (``sigma_clipping``): one scale for all the bins,
three scales from the median of the residuals. The established form is fitted with the clipping
first; every form of the block is then fitted through the bins it kept, and an alternative replaces
the established form only beyond the threshold of ``form_threshold``: by more than 10 of
``QAIC = χ²/ĉ + 2k``. A replacing form is fitted with the clipping again through all the bins. A
form competes only with more kept bins than parameters, and a form that does not converge drops
out. When no form converges with the clipping the line is the law, as the passport always comes
out; when the line does not converge either, the block has no law, and the reason is written: too
few bins or no convergence (proposal 7).

The shift of a block is its law at the zero of its axis (``block_shift``). On an axis whose zero lies
at minus infinity of its abscissa - the density ``p → 0``, the plateau of P_sync at ``v → 0`` - it is
the horizontal asymptote of a form that has one; a form without one, the line or the saturation, has
its shift at the reference of the abscissa, ``x = 0``: ``p = 2⁻¹⁰`` or 20 words, the edge of the
measured zone. A form without a finite value at ``x = 0`` - the hyperbola or the logarithmic law
over the share of the answer - has its shift at the smallest abscissa of its bins.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.passport.laws.form_threshold import Contender, pick, quasi_criterion
from iec.passport.laws.forms.law_fit import LawFit, fit_law
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.laws.law_rows import half_width
from iec.passport.laws.point_weights import Weigh
from iec.passport.laws.sigma_clipping import ClippedFit, clipped_fit, usable

#: Reason of a block without a law: fewer bins than the line needs.
REASON_FEW_POINTS = "few_points"
#: Reason of a block without a law: no candidate converged, the line neither.
REASON_NOT_CONVERGED = "not_converged"
#: The fewest bins of a law: a line goes through two.
MIN_POINTS = 2
_LINE = LinearForm()


@dataclass(frozen=True, slots=True, eq=False)
class BlockChoice:
    """The law of a block: its form, its fit and the record of every candidate.

    ``candidates`` - per form of the block ``form``, ``converged``, ``chi2``, ``free`` (the free
    parameters) and ``criterion`` ``QAIC``; ``kept`` - the bins of the law left by the clipping, a
    mask over its bins, ``None`` - all; ``fallback`` - the line was taken because no form converged;
    ``c_hat`` - the overdispersion of the criterion; ``established`` - the established form.
    """

    form: LawForm
    fit: LawFit
    candidates: tuple[dict[str, Any], ...]
    kept: np.ndarray | None = None
    fallback: bool = False
    c_hat: float = 1.0
    established: str = ""

    @property
    def free(self) -> int:
        """Return the number of the free parameters of the law."""
        return len(self.form.parameters) - len(self.fit.held)

    def value(self, x: np.ndarray) -> np.ndarray:
        """Return the law at the abscissae ``x``."""
        return self.form.value(self.fit.params, np.asarray(x, dtype=float))

    def record(self) -> dict[str, Any]:
        """Return the choice for the passport: the forms, ``ĉ`` and the record of every form."""
        return {
            "law_form": self.form.name,
            "established_form": self.established or self.form.name,
            "c_hat": self.c_hat,
            "candidates": [dict(entry) for entry in self.candidates],
            **({"fallback": "line"} if self.fallback else {}),
        }


def choose_law(
    forms: Sequence[LawForm],
    x: np.ndarray,
    values: np.ndarray,
    weigh: Weigh,
    *,
    starts: Mapping[str, Sequence[np.ndarray]] | None = None,
) -> BlockChoice | str:
    """Choose the law of a block: its established form unless an alternative is much better.

    Args:
        forms: The forms of the block: the established one first, its alternatives after it.
        x: The abscissae of the bins.
        values: Their coefficients.
        weigh: The weighting of any subset of the bins (``point_weights``).
        starts: The starts of a form by its name instead of its own; ``None`` - the starts of every
            form.

    Returns:
        ``BlockChoice``, or the reason of a block without a law: ``REASON_FEW_POINTS`` or
        ``REASON_NOT_CONVERGED``.
    """
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size < MIN_POINTS:
        return REASON_FEW_POINTS
    given = starts or {}
    established = forms[0].name if forms else _LINE.name
    position, base = _base(forms, x, values, weigh, given)
    if base is None:
        line = clipped_fit(_LINE, x, values, weigh)
        if line is None:
            return REASON_NOT_CONVERGED
        return BlockChoice(
            _LINE, line.fit, (), line.kept, fallback=True, established=established
        )
    kept = np.flatnonzero(base.kept)
    records: list[dict[str, Any]] = []
    contenders: list[Contender] = []
    for place, form in enumerate(forms):
        if kept.size <= len(form.parameters):
            # A form with as many parameters as bins goes through them without a residual: its
            # χ² says nothing of the form, so it does not compete.
            records.append({"form": form.name, "converged": False, "reason": REASON_FEW_POINTS})
            continue
        found = (
            base.fit
            if place == position
            else fit_law(form, x[kept], values[kept], weigh(kept), starts=given.get(form.name))
        )
        if not usable(found):
            records.append({"form": form.name, "converged": False})
            continue
        contender = Contender(place, found.chi2, len(form.parameters) - len(found.held))
        contenders.append(contender)
        records.append(
            {"form": form.name, "converged": True, "chi2": contender.chi2, "free": contender.free}
        )
    first = 0 if any(item.position == 0 for item in contenders) else None
    chosen, c_hat = pick(contenders, kept.size, first)
    for contender in contenders:
        records[contender.position]["criterion"] = quasi_criterion(contender, c_hat)
    final: ClippedFit = base
    if chosen != position:
        again = clipped_fit(forms[chosen], x, values, weigh, starts=given.get(forms[chosen].name))
        if again is None:
            chosen = position
        else:
            final = again
    return BlockChoice(
        forms[chosen],
        final.fit,
        tuple(records),
        final.kept,
        c_hat=c_hat,
        established=established,
    )


def _base(
    forms: Sequence[LawForm],
    x: np.ndarray,
    values: np.ndarray,
    weigh: Weigh,
    given: Mapping[str, Sequence[np.ndarray]],
) -> tuple[int, ClippedFit | None]:
    """Return the first form of the block that converges with the clipping, and its fit.

    For a law in order it is the established form; ``(-1, None)`` - no form converges.
    """
    for position, form in enumerate(forms):
        if values.size <= len(form.parameters):
            continue
        clipped = clipped_fit(form, x, values, weigh, starts=given.get(form.name))
        if clipped is not None:
            return position, clipped
    return -1, None


def block_shift(
    form: LawForm, params: np.ndarray, at_infinity: bool, low: float | None = None
) -> float:
    """Return the shift of a block: its law at the zero of its axis.

    Args:
        form: The form of the law.
        params: Its parameters.
        at_infinity: The zero of the axis lies at minus infinity of the abscissa; then the shift is
            the horizontal asymptote of a form that has one, otherwise the law at ``x = 0``.
        low: The smallest abscissa of the bins: the shift of a form that has no finite value at
            ``x = 0`` - the hyperbola and the logarithmic law over the share of the answer.
    """
    if at_infinity:
        limit = form.limit(params)
        if limit is not None and math.isfinite(limit):
            return float(limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(form.value(params, np.array([0.0]))[0])
    if not math.isfinite(value) and low is not None:
        value = float(form.value(params, np.array([low]))[0])
    return value


@dataclass(frozen=True, slots=True, eq=False)
class BlockLaw:
    """The law of a block with the rows of its parameters over the coefficients of the fit.

    ``choice`` - the chosen form and its fit; ``rows`` - the derivative of the parameters by the
    coefficients ``(parameters, coefficients)``, zero rows for the held ones; ``bins`` - the bins of
    the law; ``x`` - their abscissae; ``at_infinity`` - the zero of the axis lies at minus infinity of
    the abscissa (``block_shift``).
    """

    choice: BlockChoice
    rows: np.ndarray
    bins: tuple[str, ...]
    x: np.ndarray
    at_infinity: bool

    @property
    def form(self) -> LawForm:
        """Return the form of the law."""
        return self.choice.form

    @property
    def params(self) -> np.ndarray:
        """Return the parameters of the law."""
        return self.choice.fit.params

    @property
    def kept(self) -> np.ndarray:
        """Return the bins of the law left by the clipping, a mask over ``bins``."""
        if self.choice.kept is None:
            return np.ones(len(self.bins), dtype=bool)
        return np.asarray(self.choice.kept, dtype=bool)

    @property
    def kept_bins(self) -> tuple[str, ...]:
        """Return the names of the bins the law goes through."""
        return tuple(name for name, keep in zip(self.bins, self.kept, strict=True) if keep)

    @property
    def outliers(self) -> tuple[str, ...]:
        """Return the names of the bins clipped from the law."""
        return tuple(name for name, keep in zip(self.bins, self.kept, strict=True) if not keep)

    def value(self, x: np.ndarray | float, params: np.ndarray | None = None) -> np.ndarray:
        """Return the law at ``x``, with its own parameters or with ``params``."""
        used = self.params if params is None else params
        return self.form.value(used, np.atleast_1d(np.asarray(x, dtype=float)))

    def shift(self, params: np.ndarray | None = None) -> float:
        """Return the shift of the block with its own parameters or with ``params``."""
        used = self.params if params is None else params
        mask = self.choice.kept
        if mask is None or not np.any(mask):
            low = float(self.x.min())
        else:
            low = float(self.x[np.asarray(mask, dtype=bool)].min())
        return block_shift(self.form, used, self.at_infinity, low)


def unit_rows(columns: Sequence[int], size: int) -> np.ndarray:
    """Return the rows of points that are coefficients of the fit: ``(points, coefficients)``.

    Args:
        columns: The column of every point among the coefficients; ``-1`` - a reference bin without a
            coefficient, whose zero carries no error.
        size: The number of the coefficients of the fit.
    """
    rows = np.zeros((len(columns), size))
    for point, column in enumerate(columns):
        if column >= 0:
            rows[point, column] = 1.0
    return rows


def block_law(
    choice: BlockChoice,
    point_rows: np.ndarray,
    bins: Sequence[str],
    x: np.ndarray,
    at_infinity: bool,
) -> BlockLaw:
    """Return the law of a block with its rows over the coefficients of the fit.

    Args:
        choice: The chosen law.
        point_rows: The derivative of every point of the law by the coefficients: unit rows for the
            bins (``unit_rows``), the rows of the levels of the slices for the plateau of a grid.
        bins: The bins of the law in the order of its points.
        x: Their abscissae.
        at_infinity: The zero of the axis lies at minus infinity of the abscissa.
    """
    rows = choice.fit.sensitivity @ point_rows
    return BlockLaw(choice, rows, tuple(bins), np.asarray(x, dtype=float), at_infinity)


class LawStack:
    """The parameters of the laws of a quantity as one vector with their rows over the coefficients.

    The delta method of a quantity derived from the laws - a shift, the level, a capacity - takes the
    derivative of the quantity by every parameter by central differences and carries it to the
    coefficients through the rows. The parts are named: the blocks of the laws and the parameters of
    the grid that its own fit gives.
    """

    def __init__(self, parts: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Stack the parts: every part is its parameters and their rows over the coefficients."""
        self._slices: dict[str, slice] = {}
        start = 0
        for key, (params, _) in parts.items():
            self._slices[key] = slice(start, start + len(params))
            start += len(params)
        self.vector = np.concatenate(
            [np.asarray(params, dtype=float) for params, _ in parts.values()]
        )
        self.rows = np.vstack([np.asarray(rows, dtype=float) for _, rows in parts.values()])

    def part(self, vector: np.ndarray, key: str) -> np.ndarray:
        """Return the parameters of a part from a vector of the stack."""
        return vector[self._slices[key]]

    def gradient(self, function: Callable[[np.ndarray], float]) -> np.ndarray:
        """Return the derivative of a function of the vector by the coefficients of the fit."""
        center = self.vector
        by_parameter = np.zeros(center.size)
        for column in range(center.size):
            step = 1e-5 * max(abs(float(center[column])), 1.0)
            unit = np.zeros(center.size)
            unit[column] = step
            by_parameter[column] = (function(center + unit) - function(center - unit)) / (
                2.0 * step
            )
        result: np.ndarray = by_parameter @ self.rows
        return result


def law_entry(law: BlockLaw, joint: np.ndarray, symbol: str) -> dict[str, Any]:
    """Return the law of a block for the passport.

    Args:
        law: The law of the block.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        symbol: The symbol of the abscissa in the formula, such as ``log₂(p/2⁻¹⁰)``.

    Returns:
        ``law_form``, ``established_form``, ``c_hat``, ``formula``, ``params`` and their ``half`` by
        the names of the form, ``held``, ``chi2``, ``n_points`` (the bins kept), ``outliers`` (the
        bins clipped), ``free`` and the record of the candidates.
    """
    names = law.form.parameters
    return {
        **law.choice.record(),
        "formula": law.form.formula(symbol),
        "params": {name: float(value) for name, value in zip(names, law.params, strict=True)},
        "half": {name: half_width(law.rows[column], joint) for column, name in enumerate(names)},
        "held": [names[column] for column in law.choice.fit.held],
        "chi2": float(law.choice.fit.chi2),
        "n_points": len(law.kept_bins),
        "outliers": list(law.outliers),
        "free": law.choice.free,
    }


def form_parameters(entry: Mapping[str, Any], form: LawForm) -> np.ndarray:
    """Return the parameters of a published law in the order of its form."""
    return np.array([float(entry["params"][name]) for name in form.parameters])
