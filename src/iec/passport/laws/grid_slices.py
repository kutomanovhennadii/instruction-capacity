"""The law of the length slices of a grid: a form of the collection with its own level in every slice.

The cells of a length slice lie on a curve over ``ℓ = log₂ s``; the slices share the curve and every
slice owns its level ``μ``, the horizontal asymptote of the curve at ``s → 0``. The curve is a form of
the collection (``laws.forms``) whose parameter ``μ`` is that asymptote: the two-asymptote hyperbola,
the slanted hyperbola, the binary exponential. ``SlicedForm`` turns such a form into a form over the
cells - its parameters are the shared parameters of the curve and the levels of the slices, its
abscissa is a row ``(ℓ, slice)`` - and the cells go through the one fit of every law
(``laws.forms.law_fit``). The weights are ``1/(σᵢ² + τ²)``: the own variance of a cell and the
scatter of the cells beyond their own variances around the candidate with the smallest ``χ²`` per
degree of freedom (``laws.point_weights.dersimonian_laird``). The mutual dependence of the cells
stays out of the weights, and the errors are still carried by the full covariance through the
sensitivity. The form of the slices is chosen as the form of every block (``laws.form_threshold``):
every candidate goes through the same cells with the same weights, and the smallest ``QAIC`` wins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.block_choice import REASON_FEW_POINTS, REASON_NOT_CONVERGED
from iec.passport.laws.form_threshold import Contender, pick, quasi_criterion
from iec.passport.laws.forms.law_fit import LawFit, LawWeighting, fit_law, variance_weighting
from iec.passport.laws.forms.law_form import LAW_FORM_INVALID, LawForm
from iec.passport.laws.law_rows import half_width
from iec.passport.laws.point_weights import dersimonian_laird
from iec.passport.laws.sigma_clipping import usable

#: The parameter of a form that a slice owns: the horizontal asymptote of the curve at ``s → 0``.
SLICE_LEVEL = "mu"
#: The smallest share of its level in the derivative of a cell: below it at every cell of a slice
#: the cells lie on the slope, and the level of the slice is not set by the data.
MIN_LEVEL_SHARE = 0.05
#: The reason of a slice whose level its cells do not see.
SLICE_UNSEEN = "all cells on the slope: the level of the slice is not seen"


@dataclass(frozen=True, slots=True)
class SlicedForm:
    """A form of the collection over the cells of ``count`` slices: shared parameters, own levels.

    The abscissa is a row ``(ℓ, slot)`` per cell, the slots running from 0 to ``count − 1``.

    Raises:
        ContractViolationError: ``LAW_FORM_INVALID`` - the form has no level ``μ``.
    """

    base: LawForm
    count: int

    def __post_init__(self) -> None:
        """Check that the form has the level a slice owns."""
        if SLICE_LEVEL not in self.base.parameters:
            raise ContractViolationError(
                f"the form '{self.base.name}' has no level '{SLICE_LEVEL}': "
                "it cannot be the law of slices",
                error_code=LAW_FORM_INVALID,
            )

    @property
    def level(self) -> int:
        """Return the position of the level among the parameters of the form."""
        return self.base.parameters.index(SLICE_LEVEL)

    @property
    def shared(self) -> tuple[int, ...]:
        """Return the positions of the shared parameters among the parameters of the form."""
        return tuple(column for column in range(len(self.base.parameters)) if column != self.level)

    @property
    def name(self) -> str:
        """Return the name of the form."""
        return self.base.name

    @property
    def parameters(self) -> tuple[str, ...]:
        """Return the shared parameters, then the levels ``mu_0`` … of the slices."""
        names = self.base.parameters
        return (
            *(names[column] for column in self.shared),
            *(f"{SLICE_LEVEL}_{slot}" for slot in range(self.count)),
        )

    @property
    def held_candidates(self) -> tuple[int, ...]:
        """Return the shared parameters that the data may push to a bound of their range."""
        return tuple(
            self.shared.index(column) for column in self.base.held_candidates if column != self.level
        )

    def base_params(self, params: np.ndarray, slot: int) -> np.ndarray:
        """Return the parameters of the form for one slice: the shared ones and its level."""
        full = np.empty(len(self.base.parameters))
        full[list(self.shared)] = params[: len(self.shared)]
        full[self.level] = params[len(self.shared) + slot]
        return full

    def value(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the law at the cells: every slice its own curve."""
        slots = x[:, 1].astype(int)
        result = np.empty(len(x))
        for slot in np.unique(slots):
            mask = slots == slot
            result[mask] = self.base.value(self.base_params(params, int(slot)), x[mask, 0])
        return result

    def jacobian(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the derivatives at the cells: the shared columns and the level of the slice."""
        slots = x[:, 1].astype(int)
        width = len(self.shared)
        result = np.zeros((len(x), width + self.count))
        for slot in np.unique(slots):
            rows = np.flatnonzero(slots == slot)
            columns = self.base.jacobian(self.base_params(params, int(slot)), x[rows, 0])
            result[np.ix_(rows, np.arange(width))] = columns[:, list(self.shared)]
            result[rows, width + int(slot)] = columns[:, self.level]
        return result

    def limit(self, params: np.ndarray) -> float | None:
        """Return ``None``: the slices have no single asymptote."""
        del params
        return None

    def formula(self, symbol: str) -> str:
        """Return the formula of the curve of a slice."""
        return self.base.formula(symbol)

    def bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the range of the form for the shared parameters and for every level."""
        lower, upper = self.base.bounds(x[:, 0])
        shared = list(self.shared)
        return (
            np.concatenate([lower[shared], np.full(self.count, lower[self.level])]),
            np.concatenate([upper[shared], np.full(self.count, upper[self.level])]),
        )

    def starts(self, x: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return the starts of the form over all the cells, every level moved to the top of its slice.

        A start of the form sets the shared parameters; the level of a slice starts at the level of
        the start moved by the distance between the top cell of the slice and the top cell of all.
        """
        slots = x[:, 1].astype(int)
        top = float(values.max())
        tops = np.array([float(values[slots == slot].max()) for slot in range(self.count)])
        shared = list(self.shared)
        return tuple(
            np.concatenate([start[shared], start[self.level] + tops - top])
            for start in self.base.starts(x[:, 0], values)
        )


@dataclass(frozen=True, slots=True, eq=False)
class SliceLaw:
    """The law of the slices: the chosen form over the cells, its fit and the record of every form."""

    sliced: SlicedForm
    fit: LawFit
    candidates: tuple[dict[str, Any], ...]
    c_hat: float
    established: str

    @property
    def form(self) -> LawForm:
        """Return the form of the curve of a slice."""
        return self.sliced.base

    @property
    def width(self) -> int:
        """Return the number of the shared parameters."""
        return len(self.sliced.shared)

    @property
    def shared(self) -> np.ndarray:
        """Return the shared parameters in the order of the form without the level."""
        return np.asarray(self.fit.params[: self.width], dtype=float)

    @property
    def levels(self) -> np.ndarray:
        """Return the levels of the slices by slot."""
        return np.asarray(self.fit.params[self.width :], dtype=float)

    @property
    def shared_names(self) -> tuple[str, ...]:
        """Return the names of the shared parameters."""
        return self.sliced.parameters[: self.width]

    @property
    def held_shared(self) -> tuple[int, ...]:
        """Return the shared parameters held on a bound of their range."""
        return tuple(column for column in self.fit.held if column < self.width)

    @property
    def free(self) -> int:
        """Return the number of the free parameters."""
        return len(self.fit.params) - len(self.fit.held)

    def curve(self, level: float, ell: np.ndarray) -> np.ndarray:
        """Return the curve of a slice of a given level at ``ℓ``."""
        return slice_value(self.form, self.shared, level, ell)

    def level_shares(self, x: np.ndarray) -> np.ndarray:
        """Return the share of the level of its slice in the derivative of every cell."""
        slots = x[:, 1].astype(int)
        jacobian = self.sliced.jacobian(self.fit.params, x)
        result: np.ndarray = jacobian[np.arange(len(x)), self.width + slots]
        return result

    def rows(self, columns: Sequence[int], size: int) -> np.ndarray:
        """Return the derivative of the parameters by all the coefficients of the fit.

        Args:
            columns: The column of every cell among the coefficients.
            size: The number of the coefficients.
        """
        result = np.zeros((len(self.fit.params), size))
        result[:, list(columns)] = self.fit.sensitivity
        return result


def slice_value(form: LawForm, shared: np.ndarray, level: float, ell: np.ndarray) -> np.ndarray:
    """Return the curve of a form with the shared parameters and a level at ``ℓ``."""
    names = form.parameters
    position = names.index(SLICE_LEVEL)
    params = np.empty(len(names))
    params[[column for column in range(len(names)) if column != position]] = shared
    params[position] = level
    return form.value(params, np.atleast_1d(np.asarray(ell, dtype=float)))


def published_shared(entry: Mapping[str, Any], form: LawForm) -> np.ndarray:
    """Return the shared parameters of a published law of the slices in the order of its form."""
    return np.array([float(entry["params"][name]) for name in form.parameters if name != SLICE_LEVEL])


def _scatter(
    forms: Sequence[LawForm], x: np.ndarray, values: np.ndarray, variances: np.ndarray, count: int
) -> float:
    """Return ``τ²`` of the cells around the candidate with the smallest ``χ²`` per degree of freedom.

    The candidates go through the cells with the weights of their own variances; a cell set that no
    candidate goes through has no scatter.
    """
    own = variance_weighting(variances)
    best: tuple[float, SlicedForm, LawFit] | None = None
    for form in forms:
        sliced = SlicedForm(form, count)
        free_points = len(values) - len(sliced.parameters)
        if free_points <= 0:
            continue
        found = fit_law(sliced, x, values, own)
        if not usable(found):
            continue
        ratio = found.chi2 / (free_points + len(found.held))
        if best is None or ratio < best[0]:
            best = (ratio, sliced, found)
    if best is None:
        return 0.0
    _, sliced, found = best
    columns = [column for column in range(len(sliced.parameters)) if column not in found.held]
    return dersimonian_laird(sliced.jacobian(found.params, x)[:, columns], found.chi2, variances)


def choose_slice_form(
    forms: Sequence[LawForm], x: np.ndarray, values: np.ndarray, variances: np.ndarray, count: int
) -> SliceLaw | str:
    """Choose the form of the slices: every candidate through the same cells, the smallest QAIC wins.

    Args:
        forms: The candidate forms of the slices, the established one first.
        x: The rows ``(ℓ, slot)`` of the cells.
        values: The coefficients of the cells.
        variances: Their own variances.
        count: The number of the slices.

    Returns:
        ``SliceLaw``, or ``REASON_NOT_CONVERGED`` when no candidate converges.
    """
    own = np.asarray(variances, dtype=float)
    weighting: LawWeighting = variance_weighting(own + _scatter(forms, x, values, own, count))
    records: list[dict[str, Any]] = []
    contenders: list[Contender] = []
    fits: dict[int, tuple[SlicedForm, LawFit]] = {}
    for position, form in enumerate(forms):
        sliced = SlicedForm(form, count)
        if len(values) <= len(sliced.parameters):
            records.append({"form": form.name, "converged": False, "reason": REASON_FEW_POINTS})
            continue
        found = fit_law(sliced, x, values, weighting)
        if not usable(found):
            records.append({"form": form.name, "converged": False})
            continue
        contender = Contender(
            position, found.chi2, len(sliced.parameters) - len(found.held), bool(found.held)
        )
        contenders.append(contender)
        fits[position] = (sliced, found)
        records.append(
            {"form": form.name, "converged": True, "chi2": contender.chi2, "free": contender.free}
        )
    if not contenders:
        return REASON_NOT_CONVERGED
    chosen, c_hat = pick(contenders, len(values))
    for contender in contenders:
        records[contender.position]["criterion"] = quasi_criterion(contender, c_hat)
    sliced, found = fits[chosen]
    return SliceLaw(sliced, found, tuple(records), c_hat, forms[0].name)


def slice_entry(law: SliceLaw, rows: np.ndarray, joint: np.ndarray, symbol: str) -> dict[str, Any]:
    """Return the law of the slices for the passport.

    Args:
        law: The law of the slices.
        rows: The rows of the shared parameters over the coefficients of the fit.
        joint: The joint covariance of the coefficients and of the variance of the shift.
        symbol: The symbol of the abscissa in the formula.

    Returns:
        ``law_form``, ``established_form``, ``c_hat``, the record of the candidates, ``formula``, the
        shared ``params`` and their ``half``, ``held``, ``chi2``, ``free``, ``n_points`` - the cells of
        the fit - and ``n_parameters``.
    """
    names = law.shared_names
    return {
        "law_form": law.form.name,
        "established_form": law.established,
        "c_hat": law.c_hat,
        "candidates": [dict(entry) for entry in law.candidates],
        "formula": law.form.formula(symbol),
        "params": {name: float(value) for name, value in zip(names, law.shared, strict=True)},
        "half": {name: half_width(rows[column], joint) for column, name in enumerate(names)},
        "held": [names[column] for column in law.held_shared],
        "chi2": float(law.fit.chi2),
        "free": law.free,
        "n_points": int(law.fit.sensitivity.shape[1]),
        "n_parameters": len(law.fit.params),
    }
