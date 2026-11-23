"""The standard stress test: the task at which the median request errs at the last word in half of the cases.

The test describes how a model behaves as the complexity of a task with fixed proportions of the load increases step by
step: the density of replacements ``p = 0.5`` and one rule per 50 words of the text, so that ``m`` rules give ``T = 50·m``
words and ``D = 2m`` distinct words, the length of the text and the number of rules increasing together. The complexity
increases until the combined error of the last word of the median request, ``E = 1 − P_sync·H``, reaches 50%: the rule
word lost its place or was replaced wrong. The false replacements of background words ``F`` do not enter the definition of
the point and are measured at the point found.

The logit of a quantity at a word is its logit at the zero of the prices less the prices of the loads, each the fall of its
law from the zero of its axis: length - the text length and the word number (the grid of ``P_sync`` and ``H``, the length
and the share of the answer of ``F``), density, and rules - the number of rules and, for ``H``, the place of the rule in the
list. The zero of the prices is 20 words and the start of the answer, the density at the shift of its law (its asymptote at
``p → 0`` or its value at ``p = 2⁻¹⁰``; the top of its law for ``F``), one rule and the first place of the list. A word of
the text belongs to every rule with the same chance, so its price of rules is the mean price of the places over the list.

At the point the section gives the task in words and rules; the structure of the errors of the last word per 100 combined
errors - synchronization ``1 − P_sync``, replacements ``P_sync·(1 − H)`` and, over them, the misfires on background words ``P_sync·F``;
for every kind its errors over the text, the mean over the words of the text for the median request, and the influence of
length, density and rules per 100 errors of the kind at the last word - the shares of their prices in its logit; and the
combined error over the text. A share is negative when the quality grows with the load: the load lowers the errors of
the kind, and the three shares still sum to 100; when the prices of a kind sum to zero or less, the kind has no more errors
at the point than at the zero of the prices, and its shares are not defined. The combined error grows with the number of
rules: the
search doubles the number of rules until the last word passes one half and bisects the last doubling, and the numbers are
interpolated linearly between the two whole numbers of rules around one half.

The laws come from the published sections and their ``law_covariance``. The half-widths of the 95% intervals go by the
delta method over the covariance of every quantity, the three quantities taken independent: their fits are separate, and
the passport holds no covariance between them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.block_choice import block_shift
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.laws.forms.law_form import LawForm
from iec.passport.laws.gls_tools import STATUS_EXTRAPOLATED, STATUS_MEASURED
from iec.passport.laws.grid_slices import slice_value
from iec.passport.laws.law_rows import HALF

#: The density of the replacements of the task.
DENSITY = 0.5
#: Words of the text per rule.
WORDS_PER_RULE = 50
#: The combined error of the last word at the point.
POINT_ERROR = 0.5
#: The density on the abscissa of the laws of the density, ``log₂(p/2⁻¹⁰)``.
X_DENSITY = math.log2(DENSITY) + 10.0
#: The text length at the zero of the price of length.
REFERENCE_WORDS = 20.0
#: The largest number of rules of the search.
RULES_CEILING = 2**20
#: The most points of a grid of the words of the text or of the places of the list.
GRID_POINTS = 500
#: The quantities of the test.
QUANTITIES = ("P_sync", "H", "F")
#: The kinds of errors in the order of the quantities.
KINDS = ("synchronization", "replacements", "background")
#: The parts of the laws of every quantity: the part in ``law_covariance``, the law in ``laws`` and its block in the
#: assignment of forms.
PARTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "P_sync": (
        ("slices", "grid", "grid_slices"),
        ("plateau", "grid_plateau", "grid_plateau"),
        ("density", "density", "density"),
        ("table", "table_size", "table_size"),
    ),
    "H": (
        ("slices", "grid", "grid_slices"),
        ("plateau", "grid_plateau", "grid_plateau"),
        ("table", "table_size", "table_size"),
        ("density", "density", "density"),
        ("rule", "rule_position", "rule_position"),
    ),
    "F": (
        ("V", "V", "length"),
        ("q", "q", "answer_share"),
        ("P", "P", "density"),
        ("M", "M", "table_size"),
    ),
}
#: The numbers of the test in the order of ``_numbers``.
NAMES = (
    "words",
    "rules",
    "structure.synchronization",
    "structure.replacements",
    "structure.background",
    "synchronization.text_errors",
    "synchronization.length",
    "synchronization.density",
    "synchronization.rules",
    "replacements.text_errors",
    "replacements.length",
    "replacements.density",
    "replacements.rules",
    "background.text_errors",
    "background.length",
    "background.density",
    "background.rules",
    "combined_text_errors",
)
#: The last word fails in half of the cases already at one rule.
STATUS_BELOW_ONE_RULE = "below_one_rule"
#: The last word does not fail in half of the cases up to ``RULES_CEILING`` rules.
STATUS_BEYOND_RANGE = "beyond_range"
#: A quantity of the passport has no laws.
STATUS_NO_LAWS = "no_laws"
#: Error code: the assignment of forms has no form of a published law.
STRESS_FORM_UNKNOWN = "STRESS_FORM_UNKNOWN"


def _logistic(x: np.ndarray | float) -> np.ndarray:
    """Return the logistic function, ``0.5·(1 + tanh(x/2))``: no overflow at any logit."""
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float))))


def _grid(top: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the points of the whole numbers ``1…top`` and the share of them every point stands for.

    Up to ``GRID_POINTS`` numbers every number is a point; above, the points go evenly on the logarithmic scale.
    """
    if top <= GRID_POINTS:
        return np.arange(1, top + 1, dtype=float), np.full(top, 1.0 / top)
    points = np.unique(np.round(np.geomspace(1.0, float(top), GRID_POINTS)))
    edges = np.concatenate(([0.5], 0.5 * (points[1:] + points[:-1]), [top + 0.5]))
    return points, np.diff(edges) / top


class _Quantity:
    """The laws of one quantity from its published section: its zero logit and the prices of the loads at the words."""

    def __init__(
        self, quantity: str, section: Mapping[str, Any], assignment: LawAssignment
    ) -> None:
        """Read the laws of a quantity: their forms, the parts of the parameter vector and its covariance.

        Args:
            quantity: ``P_sync``, ``H`` or ``F``.
            section: The published section of the quantity with ``law_covariance``.
            assignment: The forms of the laws of the passport.

        Raises:
            ContractViolationError: ``STRESS_FORM_UNKNOWN`` - the assignment has no form of a published law.
        """
        entry = section["law_covariance"]
        self.quantity = quantity
        self.vector = np.asarray(entry["vector"], dtype=float)
        self.covariance = np.asarray(entry["matrix"], dtype=float)
        self._parts: dict[str, slice] = {}
        start = 0
        for part in entry["parts"]:
            self._parts[part["part"]] = slice(start, start + len(part["names"]))
            start += len(part["names"])
        self._forms: dict[str, LawForm] = {}
        for key, law_key, block in PARTS[quantity]:
            name = section["laws"][law_key]["law_form"]
            form = next(
                (item for item in assignment.forms_of(quantity, block) if item.name == name), None
            )
            if form is None:
                raise ContractViolationError(
                    f"the assignment of forms has no form '{name}' of {quantity}.{block}",
                    error_code=STRESS_FORM_UNKNOWN,
                )
            self._forms[key] = form
        self._x_top = (
            math.log2(float(section["laws"]["P"]["p_top"])) + 10.0 if quantity == "F" else 0.0
        )

    def _law(self, vector: np.ndarray, key: str, x: np.ndarray | float) -> np.ndarray:
        """Return the law of a part at the abscissae ``x``."""
        values: np.ndarray = self._forms[key].value(
            vector[self._parts[key]], np.atleast_1d(np.asarray(x, dtype=float))
        )
        return values

    def _at(self, vector: np.ndarray, key: str, x: float) -> float:
        """Return the law of a part at one abscissa."""
        return float(self._law(vector, key, x)[0])

    def prices(
        self, vector: np.ndarray, words: np.ndarray, m: int
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Return the zero logit and the prices of the loads at the words of the task of ``m`` rules.

        Args:
            vector: The parameters of the laws of the quantity.
            words: The numbers of the words of the text.
            m: The number of rules; the text has ``50·m`` words.

        Returns:
            ``(zero, length, rules, shares, density)``: the logit of the median request at the zero of the prices; the
            price of length at every word; the price of rules at every place of the list taken and the share of the list
            it stands for; the price of density.
        """
        v = float(WORDS_PER_RULE * m)
        if self.quantity == "F":
            v_zero = self._at(vector, "V", 0.0)
            q_zero = self._at(vector, "q", 0.0)
            p_zero = self._at(vector, "P", self._x_top)
            m_zero = self._at(vector, "M", 0.0)
            length = (v_zero - self._at(vector, "V", math.log2(v / REFERENCE_WORDS))) + (
                q_zero - self._law(vector, "q", words / v)
            )
            rules = np.array([m_zero - self._at(vector, "M", math.log2(m))])
            density = p_zero - self._at(vector, "P", X_DENSITY)
            return v_zero + q_zero + p_zero + m_zero, length, rules, np.ones(1), density
        u = max(math.log2(v / REFERENCE_WORDS), 0.0)
        plateau_zero = self._at(vector, "plateau", 0.0)
        level = self._at(vector, "plateau", u)
        length = plateau_zero - slice_value(
            self._forms["slices"], vector[self._parts["slices"]], level, np.log2(words)
        )
        density_zero = block_shift(self._forms["density"], vector[self._parts["density"]], True)
        density = density_zero - self._at(vector, "density", X_DENSITY)
        table_zero = self._at(vector, "table", 0.0)
        count = table_zero - self._at(vector, "table", math.log2(m))
        zero = plateau_zero + density_zero + table_zero
        if self.quantity == "P_sync":
            return zero, length, np.array([count]), np.ones(1), density
        places, shares = _grid(m)
        rule_zero = self._at(vector, "rule", 0.0)
        rules = count + (rule_zero - self._law(vector, "rule", np.log2(places)))
        return zero + rule_zero, length, rules, shares, density

    def last(self, vector: np.ndarray, m: int) -> tuple[float, np.ndarray]:
        """Return the success of the last word of the text and the prices of length, rules and density there.

        The logit of the word takes the mean price of the places of the rules over the list.
        """
        zero, length, rules, shares, density = self.prices(
            vector, np.array([float(WORDS_PER_RULE * m)]), m
        )
        prices = np.array([float(length[0]), float(rules @ shares), density])
        return float(_logistic(zero - prices.sum())), prices

    def text_success(self, vector: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the success at the points of the words of the text, mean over the places of the list, and their shares."""
        words, weights = _grid(WORDS_PER_RULE * m)
        zero, length, rules, shares, density = self.prices(vector, words, m)
        success = _logistic(zero - (length[:, None] + density + rules[None, :])) @ shares
        return np.asarray(success, dtype=float), weights


def _combined_last(
    models: Mapping[str, _Quantity], vectors: Mapping[str, np.ndarray], m: int
) -> float:
    """Return the combined error of the last word of the task of ``m`` rules."""
    return (
        1.0 - models["P_sync"].last(vectors["P_sync"], m)[0] * models["H"].last(vectors["H"], m)[0]
    )


def _bracket(
    models: Mapping[str, _Quantity], vectors: Mapping[str, np.ndarray]
) -> tuple[int, int, float] | str:
    """Return the whole numbers of rules around the point and the fraction between them, or a status without a point."""
    low_error = _combined_last(models, vectors, 1)
    if low_error >= POINT_ERROR:
        return STATUS_BELOW_ONE_RULE
    low, high = 1, 2
    high_error = _combined_last(models, vectors, high)
    while high_error < POINT_ERROR:
        if high >= RULES_CEILING:
            return STATUS_BEYOND_RANGE
        low, low_error = high, high_error
        high *= 2
        high_error = _combined_last(models, vectors, high)
    while high - low > 1:
        middle = (low + high) // 2
        error = _combined_last(models, vectors, middle)
        if error < POINT_ERROR:
            low, low_error = middle, error
        else:
            high, high_error = middle, error
    return low, high, (POINT_ERROR - low_error) / (high_error - low_error)


def _reading(
    models: Mapping[str, _Quantity], vectors: Mapping[str, np.ndarray], m: int
) -> np.ndarray:
    """Return the raw numbers of the task of ``m`` rules.

    The structure of the last word, ``1 − P_sync``, ``P_sync·(1 − H)``, ``P_sync·F``; the errors over the text of
    the three kinds; the combined error over the text; the shares of the prices of length, rules and density at the last
    word of every kind.
    """
    lasts = [models[quantity].last(vectors[quantity], m) for quantity in QUANTITIES]
    (sync, _), (rule, _), (background, _) = lasts
    structure = [1.0 - sync, sync * (1.0 - rule), sync * (1.0 - background)]
    texts = [models[quantity].text_success(vectors[quantity], m) for quantity in QUANTITIES]
    weights = texts[0][1]
    errors = [float((1.0 - success) @ weights) for success, _ in texts]
    combined = float((1.0 - texts[0][0] * texts[1][0]) @ weights)
    shares: list[float] = []
    for _, prices in lasts:
        total = float(prices.sum())
        shares += [float(price) / total if total > 0.0 else math.nan for price in prices]
    return np.array([*structure, *errors, combined, *shares], dtype=float)


def _numbers(
    models: Mapping[str, _Quantity], vectors: Mapping[str, np.ndarray]
) -> np.ndarray | str:
    """Return the numbers of the test at the point in the order of ``NAMES``, or a status without a point."""
    found = _bracket(models, vectors)
    if isinstance(found, str):
        return found
    low, high, fraction = found
    raw = (1.0 - fraction) * _reading(models, vectors, low) + fraction * _reading(
        models, vectors, high
    )
    structure, errors, combined, shares = raw[0:3], raw[3:6], raw[6], raw[7:16].reshape(3, 3)
    rules = low + fraction
    numbers = [WORDS_PER_RULE * rules, rules, *(100.0 * structure / (structure[0] + structure[1]))]
    for kind in range(3):
        length, rule_share, density = shares[kind]
        numbers += [100.0 * errors[kind], 100.0 * length, 100.0 * density, 100.0 * rule_share]
    numbers.append(100.0 * combined)
    return np.array(numbers, dtype=float)


def _gradient(function: Callable[[np.ndarray], np.ndarray], center: np.ndarray) -> np.ndarray:
    """Return the derivatives of the numbers by the parameters by central differences: ``(numbers, parameters)``."""
    columns = []
    for column in range(center.size):
        step = 1e-5 * max(abs(float(center[column])), 1.0)
        unit = np.zeros(center.size)
        unit[column] = step
        columns.append((function(center + unit) - function(center - unit)) / (2.0 * step))
    return np.array(columns).T


def _support(section: Mapping[str, Any]) -> dict[str, float]:
    """Return the edges of the measured area: the longest text and the largest number of rules of the bins of ``H``."""
    bins = section["bins"]
    words = max(float(cell["words"][1]) for cell in bins["grid"] if cell["requests"] > 0)
    rules = 2.0 ** (max(int(row["bin"]) for row in bins["M"] if row["requests"] > 0) - 1)
    return {"words": words, "rules": rules}


def _row(value: float, half: float) -> dict[str, float | None]:
    """Return a number of the test with the half-width of its 95% interval; what a float does not hold is ``None``."""
    return {
        "value": value if math.isfinite(value) else None,
        "half": half if math.isfinite(half) else None,
    }


def stress_test_section(passport: Mapping[str, Any], assignment: LawAssignment) -> dict[str, Any]:
    """Return the section ``stress_test`` of a passport from its three published sections.

    Args:
        passport: The passport with the sections ``P_sync``, ``H`` and ``F``; a section with laws has ``law_covariance``.
        assignment: The forms of the laws of the passport (``config/passport_laws.yaml``).

    Returns:
        ``protocol`` and ``status``; with a point also ``support``, ``words``, ``rules``, ``combined_error_at_point``,
        ``structure``, the kinds ``synchronization``, ``replacements`` and ``background`` with ``text_errors`` and
        ``influence``, and ``combined_text_errors``; every number is ``{"value", "half"}`` in words, rules or percent.
        The status of a point is ``measured`` inside the measured area of ``support`` and ``extrapolated`` past it;
        without a point it is ``below_one_rule``, ``beyond_range`` or ``no_laws``.
    """
    protocol = {"density": DENSITY, "words_per_rule": WORDS_PER_RULE, "point_error": POINT_ERROR}
    sections = [passport.get(quantity, {}) for quantity in QUANTITIES]
    if any(
        section.get("status") != STATUS_MEASURED or "law_covariance" not in section
        for section in sections
    ):
        return {"protocol": protocol, "status": STATUS_NO_LAWS}
    models = {
        quantity: _Quantity(quantity, passport[quantity], assignment) for quantity in QUANTITIES
    }
    vectors = {quantity: model.vector for quantity, model in models.items()}
    values = _numbers(models, vectors)
    if isinstance(values, str):
        return {"protocol": protocol, "status": values}
    variance = np.zeros(len(NAMES))
    for quantity, model in models.items():

        def moved(vector: np.ndarray, quantity: str = quantity) -> np.ndarray:
            """Return the numbers of the test with the parameters of one quantity moved."""
            numbers = _numbers(models, {**vectors, quantity: vector})
            return np.full(len(NAMES), math.nan) if isinstance(numbers, str) else numbers

        gradient = _gradient(moved, model.vector)
        variance += np.einsum("ij,jk,ik->i", gradient, model.covariance, gradient)
    halves = HALF * np.sqrt(np.maximum(variance, 0.0))
    entry = {
        name: _row(float(value), float(half))
        for name, value, half in zip(NAMES, values, halves, strict=True)
    }
    support = _support(passport["H"])
    inside = bool(values[1] <= support["rules"] and values[0] <= support["words"])

    def kind(name: str) -> dict[str, Any]:
        """Return the numbers of one kind of errors."""
        return {
            "text_errors": entry[f"{name}.text_errors"],
            "influence": {load: entry[f"{name}.{load}"] for load in ("length", "density", "rules")},
        }

    return {
        "protocol": protocol,
        "status": STATUS_MEASURED if inside else STATUS_EXTRAPOLATED,
        "support": support,
        "words": entry["words"],
        "rules": entry["rules"],
        "combined_error_at_point": 100.0 * POINT_ERROR,
        "structure": {name: entry[f"structure.{name}"] for name in KINDS},
        **{name: kind(name) for name in KINDS},
        "combined_text_errors": entry["combined_text_errors"],
    }
