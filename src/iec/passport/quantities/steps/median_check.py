"""The check of the variance by the load against the median request of the data.

The GLMM with heterogeneous random-intercept variance lets the variance of the request shift depend
on the load. A bin of the passport is the logit of the median request of the model; where the spread
of the requests changes along an axis, the bin can lie on the other side of the reference bin than
the requests themselves. Every bin of an axis constant within a request - the density, the number of
rules and, for F, the text length - is compared with the median request of the data: the requests
with at least ``MIN_TRIALS`` trials of the quantity are grouped into the cells of the other two axes;
in every cell the median shares of failures of the requests of the bin and of the reference bin go
to the logit of the odds of a success, ``ln((1 − s)/s)``, and their difference is averaged over the
cells with the weight of the number of the requests of the reference bin. Its error comes from
``BOOTSTRAP_ROUNDS`` resamplings of the requests with the seed ``BOOTSTRAP_SEED``. A bin and its
comparison on the two sides of the reference bin, the comparison beyond ``HALF`` standard errors,
mean that the variance by the load turned the law against the data, and the quantity is fitted with
one variance, the regular fallback of the fit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from iec.passport.fitting.chain.fit_notes import SPREAD_AGAINST_DATA, FitNote
from iec.passport.fitting.chain.fit_paths import METHOD_GLMM_HETEROGENEOUS
from iec.passport.fitting.load_spread import RequestLoad, request_load
from iec.passport.fitting.variable_fit import VariableFit, fit_variable
from iec.passport.laws.law_rows import HALF
from iec.passport.trials.design.folded_design import VariableDesign
from iec.passport.trials.layout.bin_layout import BinLayout

#: The fewest trials of a request whose share of failures enters the comparison.
MIN_TRIALS = 50
#: The resamplings of the requests for the error of a comparison.
BOOTSTRAP_ROUNDS = 200
#: The seed of the resamplings: the check gives the same answer on the same points.
BOOTSTRAP_SEED = 0
#: The share of failures is kept inside ``[SHARE_FLOOR, 1 − SHARE_FLOOR]`` before its logit.
SHARE_FLOOR = 0.005
#: The place of an axis in the bins of a request load: length, density, number of rules.
LOAD_AXES = {"V": 0, "P": 1, "M": 2}


@dataclass(frozen=True, slots=True)
class Request:
    """A request: its bins on the three axes, its trials and failures of a quantity."""

    bins: tuple[int, int, int]
    trials: float
    failures: float


@dataclass(frozen=True, slots=True)
class AxisBins:
    """The bins of an axis of a fit: the number of the reference bin and the name of every bin."""

    reference: int
    names: Mapping[int, str]


def requests_of(
    points: Iterable[Any], layout: BinLayout, outcome: Callable[[Any], tuple[float, float]]
) -> list[Request]:
    """Return the requests of the points: the bins of their load and their trials and failures."""
    found: list[Request] = []
    for point in points:
        load: RequestLoad = request_load(layout, point.D, point.n, point.m)
        trials, failures = outcome(point)
        found.append(Request(load.bins, float(trials), float(failures)))
    return found


def axis_bins(design: VariableDesign, letters: Sequence[str]) -> dict[str, AxisBins]:
    """Return the bins of the axes of a folded design by their letters."""
    return {
        letter: AxisBins(
            design.axes[letter].reference,
            {item.number: item.name for item in design.axes[letter].bins},
        )
        for letter in letters
    }


def _logit(share: float) -> float:
    """Return the logit of the odds of a success of a share of failures kept off zero and one."""
    kept = min(max(share, SHARE_FLOOR), 1.0 - SHARE_FLOOR)
    return math.log((1.0 - kept) / kept)


def median_comparison(requests: Sequence[Request], axis: int, reference: int) -> dict[int, float]:
    """Return by bin of an axis its median request against the reference bin, in logits of a success."""
    first, second = (place for place in range(3) if place != axis)
    cells: dict[tuple[int, int], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in requests:
        cells[(item.bins[first], item.bins[second])][item.bins[axis]].append(
            item.failures / item.trials
        )
    totals: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for by_bin in cells.values():
        base = by_bin.get(reference)
        if not base:
            continue
        weight = float(len(base))
        level = _logit(float(np.median(base)))
        for number, shares in by_bin.items():
            if number != reference:
                totals[number][0] += weight * (_logit(float(np.median(shares))) - level)
                totals[number][1] += weight
    return {number: own / mass for number, (own, mass) in totals.items() if mass > 0.0}


def spread_against_data(
    requests: Sequence[Request], axes: Mapping[str, AxisBins], theta: Mapping[str, float]
) -> list[str]:
    """Return the bins whose coefficient lies against the median request beyond its error.

    Args:
        requests: The requests of the measurement.
        axes: The bins of the axes checked by their letters.
        theta: The coefficients of the fit by name.
    """
    kept = [item for item in requests if item.trials >= MIN_TRIALS]
    if not kept:
        return []
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    found: list[str] = []
    for letter, bins in axes.items():
        axis = LOAD_AXES[letter]
        center = median_comparison(kept, axis, bins.reference)
        samples: dict[int, list[float]] = defaultdict(list)
        for _ in range(BOOTSTRAP_ROUNDS):
            drawn = [kept[index] for index in generator.integers(0, len(kept), len(kept))]
            for number, value in median_comparison(drawn, axis, bins.reference).items():
                samples[number].append(value)
        for number, value in sorted(center.items()):
            name = bins.names.get(number)
            if name is None or name not in theta or len(samples[number]) < 2:
                continue
            half = HALF * float(np.std(samples[number], ddof=1))
            coefficient = float(theta[name])
            if coefficient * value < 0.0 and abs(value) > half:
                found.append(
                    f"{name}: passport {coefficient:+.2f}, median request {value:+.2f} ± {half:.2f}"
                )
    return found


def checked_fit(
    design: VariableDesign,
    run_weight: np.ndarray,
    loads: Mapping[str, RequestLoad],
    requests: Sequence[Request],
    axes: Mapping[str, AxisBins],
) -> VariableFit:
    """Fit a quantity with the variance by the load; a law turned against the data - with one variance.

    Args:
        design: The folded matrix of the quantity.
        run_weight: The weights of the requests.
        loads: The loads of the requests by their keys.
        requests: The requests with their trials and failures of the quantity.
        axes: The bins of the axes checked by their letters.

    Returns:
        The fit; a fit with one variance carries the note ``SPREAD_AGAINST_DATA`` naming the bins.
    """
    fit = fit_variable(design, run_weight, loads)
    if fit.method != METHOD_GLMM_HETEROGENEOUS:
        return fit
    flipped = spread_against_data(requests, axes, dict(zip(fit.names, fit.theta, strict=True)))
    if not flipped:
        return fit
    common = fit_variable(design, run_weight, None)
    note = FitNote(
        SPREAD_AGAINST_DATA,
        "the variance by the load turns bins against the median request: " + "; ".join(flipped),
    )
    return replace(common, notes=(*common.notes, note))
