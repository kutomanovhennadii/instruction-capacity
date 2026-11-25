"""The trials of F by the share of the answer and the folded matrix of model V-(s/v)-PM.

A trial of F is a background word standing on its place in the answer; a success - no misfire: no
right-hand word of a rule stands on the place of the word. The section of F reads the trials of a
point by ten equal bins of the share of the answer ``q = s/v`` and folds them into the matrix of its
model: bins of the text length, of the share of the answer, of the density and of the number of
rules, with the tallies of every bin - the counts, the weights of the requests with failures and a
weighted centre. The generic design of the trials (``trials.design``) folds F by the word number
for the grid model; the section of F uses this design of its own.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from iec.answer_validator import KIND_MATCHED, AlignedAnswer
from iec.contracts import RuleTable, SourceText
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.design.folded_design import VariableDesign
from iec.passport.trials.layout.axis_bins import axis_density_bin, axis_density_reference
from iec.passport.trials.layout.bin_layout import BinLayout

#: Equal bins of the share of the answer; the reference bin is the last tenth.
SHARE_BINS = 10
_WEIGHT_CODE = "POINT_WEIGHT_MISSING"


@dataclass(frozen=True, slots=True)
class BackgroundCells:
    """The trials of F of one point by the share of the answer.

    ``cells`` - ``(share bin, trials, misfires)``.
    """

    key: str
    D: int  # noqa: N815 - protocol axis name
    n: int
    m: int
    cells: tuple[tuple[int, int, int], ...]


def background_cells(
    key: str,
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    aligned: AlignedAnswer,
    source_text: SourceText,
    rule_table: RuleTable,
) -> BackgroundCells:
    """Return the trials of F of a point: background words standing on their places.

    Args:
        key: The key of the point.
        D: The number of distinct words of the text.
        n: How many times every word occurs in the text.
        m: The number of rules.
        aligned: The alignment.
        source_text: The source text of the task.
        rule_table: The rule table.
    """
    targets = frozenset(rule_table.targets)
    total = len(source_text.words)
    # A background word on its place is a matched place whose reference word no rule covers; it is
    # counted on the numeric arrays of the alignment, without building a place object per word.
    matched = aligned.kinds == KIND_MATCHED
    reference = aligned.reference_positions[matched]
    answer = aligned.answer_positions[matched]
    keep = ~np.isin(reference, np.asarray(source_text.replacement_positions, dtype=np.int64))
    reference, answer = reference[keep], answer[keep]
    share_bins = np.minimum(SHARE_BINS, ((reference + 1) / total * SHARE_BINS).astype(np.int64) + 1)
    words = aligned.core_words
    misfires = np.fromiter(
        (words[place] in targets for place in answer.tolist()), dtype=bool, count=len(answer)
    )
    trials = np.bincount(share_bins, minlength=SHARE_BINS + 1)
    failures = np.bincount(share_bins, weights=misfires, minlength=SHARE_BINS + 1)
    return BackgroundCells(
        key=key,
        D=D,
        n=n,
        m=m,
        cells=tuple(
            (int(share_bin), int(trials[share_bin]), int(failures[share_bin]))
            for share_bin in np.flatnonzero(trials)
        ),
    )


@dataclass
class BinTally:
    """An accumulator of a bin: counts, the weights of requests with failures, a weighted centre."""

    trials: float = 0.0
    failures: float = 0.0
    weighted: float = 0.0
    total: float = 0.0
    failing: dict[str, float] | None = None

    def add(self, key: str, weight: float, trials: float, failures: float, value: float) -> None:
        """Add the trials of a request to the bin.

        Args:
            key: The key of the request.
            weight: Its weight.
            trials: Its trials in the bin.
            failures: Its failures in the bin.
            value: The value of the axis of the request, averaged with the weight of its trials.
        """
        self.trials += trials
        self.failures += failures
        self.weighted += weight * trials
        self.total += weight * trials * value
        if failures > 0.0:
            if self.failing is None:
                self.failing = {}
            self.failing[key] = weight

    @property
    def requests(self) -> int:
        """Return the number of requests with failures in the bin."""
        return len(self.failing or ())

    @property
    def effective(self) -> float:
        """Return the effective number of requests with failures, ``(Σw)²/Σw²``."""
        weights = np.array(list((self.failing or {}).values()), dtype=float)
        return float(weights.sum() ** 2 / (weights * weights).sum()) if len(weights) else 0.0

    @property
    def mean(self) -> float:
        """Return the centre of the bin: the value of the axis averaged with the weighted trials."""
        return self.total / self.weighted if self.weighted > 0.0 else 0.0


@dataclass(frozen=True, slots=True, eq=False)
class AnswerShareDesign:
    """The folded matrix of model V-(s/v)-PM and the tallies of its bins by axis."""

    design: VariableDesign
    bins: dict[str, dict[str, BinTally]]
    centers: dict[str, dict[str, float]]
    references: dict[str, str]
    largest_words: int


def answer_share_design(
    points: Sequence[BackgroundCells], layout: BinLayout, weights: Mapping[str, float]
) -> AnswerShareDesign:
    """Fold the trials of F into the matrix of the model.

    A row is a point and a share bin: the conditions of the request are common to the row, and the
    folding is exact. A point without trials gives no rows (``p = 1``).

    Args:
        points: The trials of F of the points.
        layout: The bin layout of the passport.
        weights: The request weights by the cube ``V × P × M``.

    Raises:
        ContractViolationError: ``POINT_WEIGHT_MISSING`` - a point with trials has no weight.
    """
    density_reference = f"p_bin_{axis_density_reference(layout)}"
    references = {
        "q": f"ql_bin_{SHARE_BINS}",
        "P": density_reference,
        "M": "m_bin_1",
    }
    tallies: dict[str, dict[str, BinTally]] = {letter: defaultdict(BinTally) for letter in "VqPM"}
    rows: list[tuple[int, list[str], int, int]] = []
    keys: list[str] = []
    largest = 0
    for point in points:
        if not point.cells:
            continue
        if point.key not in weights:
            raise ContractViolationError(
                f"point {point.key} has no weight", error_code=_WEIGHT_CODE
            )
        weight = float(weights[point.key])
        words = point.D * point.n
        largest = max(largest, words)
        volume = f"v_bin_{layout.volume_bin(words)}"
        density = f"p_bin_{axis_density_bin(layout, point.D, point.m)}"
        rules = f"m_bin_{layout.row_bin(point.m)}"
        run = len(keys)
        keys.append(point.key)
        trials = sum(cell[1] for cell in point.cells)
        failures = sum(cell[2] for cell in point.cells)
        tallies["V"][volume].add(point.key, weight, trials, failures, math.log2(words))
        tallies["P"][density].add(point.key, weight, trials, failures, point.m / point.D)
        tallies["M"][rules].add(point.key, weight, trials, failures, math.log(point.m))
        for share_bin, count, misfires in point.cells:
            share = f"ql_bin_{share_bin}"
            tallies["q"][share].add(
                point.key, weight, count, misfires, (share_bin - 0.5) / SHARE_BINS
            )
            rows.append((run, [volume, share, density, rules], count, misfires))
    reference_names = set(references.values())
    names: list[str] = []
    for letter in "VqPM":
        ordered = sorted(tallies[letter], key=lambda name: int(name.rsplit("_", 1)[1]))
        names += [name for name in ordered if name not in reference_names]
    index = {name: column for column, name in enumerate(names)}
    x = np.zeros((len(rows), len(names)))
    for row_number, (_, active, _, _) in enumerate(rows):
        for name in active:
            column = index.get(name)
            if column is not None:
                x[row_number, column] = 1.0
    centers: dict[str, dict[str, float]] = {
        "V": {name: 2.0**tally.mean for name, tally in tallies["V"].items()},
        "q": {name: (int(name.rsplit("_", 1)[1]) - 0.5) / SHARE_BINS for name in tallies["q"]},
        "P": {name: tally.mean for name, tally in tallies["P"].items()},
        "M": {name: math.exp(tally.mean) for name, tally in tallies["M"].items()},
    }
    design = VariableDesign(
        variable="F",
        x=x,
        failures=np.array([row[3] for row in rows], dtype=float),
        trials=np.array([row[2] for row in rows], dtype=float),
        run=np.array([row[0] for row in rows], dtype=np.int64),
        point_keys=tuple(keys),
        names=tuple(names),
        cells=(),
        axes={},
        abscissae={},
    )
    return AnswerShareDesign(
        design=design,
        bins={letter: dict(value) for letter, value in tallies.items()},
        centers=centers,
        references=references,
        largest_words=largest,
    )
