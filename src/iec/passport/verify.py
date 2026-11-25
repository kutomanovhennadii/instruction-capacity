"""Verification of a published measurement: integrity and reproducibility.

Four checks, in this order: (1) the SHA-256 of every file listed in ``measurement.json``;
(2) the vocabulary - its hash equals the description, and it matches its reference version;
(3) the generator rebuilds exactly the recorded rule list and source text of every point from
its seeds; (4) the passport recomputed from the points equals the published one - the
composition exactly, the numbers within the tolerance of computation. ``verify`` checks
integrity and reproducibility only; it does not prove the method right.

**Tolerance of check 4.** Reordering the summation of the same points - a perturbation at the
level of rounding, the same a different machine makes - moved the passport of the published
measurement by at most 7.6e-9 relative in the fits and the weights, and by at most 9.5e-6 relative
in everything computed from them: the bins restored by the laws, the laws, the levels and the
capacities. The tolerances below keep a margin of about 100 times and more over these spreads. A
number near zero is compared by the absolute floor. The record of a candidate form that a law did
not choose is compared by the name of the form alone: a fit at the edge of its convergence converges
on one machine and not on another, and the chosen law and its numbers stay the same. For the same
reason the number of steps of a fit is not compared: it is the path of the optimizer, not its result,
and a fit near its stopping rule takes a step more or less on another machine with the same numbers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iec.contracts import MeasurementDescription, PointRecord, Vocab
from iec.infrastructure.exceptions import IECError
from iec.model_client import MockTokenizer
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.measurement_files import (
    points_names,
    read_description,
    read_passport_data,
    read_points,
    sha256_of,
)
from iec.passport.passport_data import build_passport_data, measurement_trials
from iec.prompt_generator import RuleGenerator, TextGenerator
from iec.vocab import VocabStore, VocabVerifier

#: Relative tolerance of the fits and the weights.
FIRST_STAGE_RTOL = 1e-6
#: Relative tolerance of the numbers computed from them.
SECOND_STAGE_RTOL = 1e-3
#: Absolute floor of every float tolerance: a difference of zeros.
ABSOLUTE_FLOOR = 1e-9

FILES_CHECK = "files"
VOCABULARY_CHECK = "vocabulary"
GENERATOR_CHECK = "generator"
PASSPORT_CHECK = "passport"

#: The keys whose numbers are the first stage, wherever they stand in the passport.
_FIRST_STAGE_KEYS = frozenset({"fit", "weighting"})
#: The key of the records of the candidate forms of a law and the key of its chosen form.
_CANDIDATES, _LAW_FORM = "candidates", "law_form"
#: The key of the number of steps of a fit: the path of the optimizer, not compared under ``fit``.
_STEPS = "steps"
#: Differences listed per check; the rest is counted.
_SHOWN = 20


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of one check: its name, whether it passed and what differs."""

    name: str
    passed: bool
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Outcomes of the four checks in their order."""

    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Return whether every check passed."""
        return bool(self.checks) and all(check.passed for check in self.checks)


def verify_measurement(
    directory: Path, vocab_dir: Path, assignment: LawAssignment, *, executor: Executor | None = None
) -> VerificationReport:
    """Run the four checks on a published measurement directory.

    Args:
        directory: The published measurement directory.
        vocab_dir: The folder of the vocabularies and their references.
        assignment: The forms of the laws of the passport.
        executor: The pool of processes of the recomputation of the passport, or ``None``.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the description is absent.
    """
    description = read_description(directory)
    files = _check_files(directory, description)
    if not files.passed:
        skipped = "not run: the files of the measurement do not match their checksums"
        return VerificationReport(
            (
                files,
                CheckResult(VOCABULARY_CHECK, False, (skipped,)),
                CheckResult(GENERATOR_CHECK, False, (skipped,)),
                CheckResult(PASSPORT_CHECK, False, (skipped,)),
            )
        )
    records = read_points(directory)
    vocabulary, vocab = _check_vocabulary(description, vocab_dir)
    generator = (
        _check_generator(records, vocab)
        if vocab is not None
        else CheckResult(GENERATOR_CHECK, False, ("not run: the vocabulary is not verified",))
    )
    passport = _check_passport(directory, description, records, assignment, executor)
    return VerificationReport((files, vocabulary, generator, passport))


def _listed(name: str, differences: Sequence[str]) -> CheckResult:
    """Return a check result with at most ``_SHOWN`` differences and the count of the rest."""
    shown = tuple(differences[:_SHOWN])
    if len(differences) > _SHOWN:
        shown += (f"... {len(differences)} difference(s) in all",)
    return CheckResult(name, not differences, shown)


def _check_files(directory: Path, description: MeasurementDescription) -> CheckResult:
    """Check 1: every listed file exists and has its SHA-256; every file of the points is listed."""
    differences: list[str] = []
    for name in points_names(directory):
        if name not in description.files:
            differences.append(f"{name} is not listed in the description")
    for name, digest in sorted(description.files.items()):
        path = directory / name
        if not path.is_file():
            differences.append(f"{name}: the file does not exist")
        elif sha256_of(path) != digest:
            differences.append(f"{name}: SHA-256 differs from the description")
    return _listed(FILES_CHECK, differences)


def _check_vocabulary(
    description: MeasurementDescription, vocab_dir: Path
) -> tuple[CheckResult, Vocab | None]:
    """Check 2: the vocabulary loads, its hash is that of the description, it matches its
    reference version (library versions are not compared)."""
    try:
        vocab = VocabStore(vocab_dir).load(description.vocab_id)
        verdict = VocabVerifier(vocab_dir).verify(description.vocab_id, check_provenance=False)
    except IECError as error:
        return CheckResult(
            VOCABULARY_CHECK, False, (f"[{error.error_code}] {error.message}",)
        ), None
    differences = list(verdict.mismatches)
    if vocab.hash != description.vocab_hash:
        differences.append(
            f"vocabulary hash {vocab.hash} differs from the description {description.vocab_hash}"
        )
    result = _listed(VOCABULARY_CHECK, differences)
    return result, (vocab if result.passed else None)


def _check_generator(records: Sequence[PointRecord], vocab: Vocab) -> CheckResult:
    """Check 3: the rule list and the source text of every point rebuild from its seeds."""
    rules = RuleGenerator(vocab)
    texts = TextGenerator(vocab, MockTokenizer())
    differences: list[str] = []
    for record in records:
        table = rules.generate(record.conditions)
        if tuple((rule.source, rule.target) for rule in table.rules) != record.rules:
            differences.append(f"{record.conditions.identity}: the rule list differs")
            continue
        if texts.generate(record.conditions, table).text != record.source_text:
            differences.append(f"{record.conditions.identity}: the source text differs")
    return _listed(GENERATOR_CHECK, differences)


def _check_passport(
    directory: Path,
    description: MeasurementDescription,
    records: Sequence[PointRecord],
    assignment: LawAssignment,
    executor: Executor | None,
) -> CheckResult:
    """Check 4: the passport recomputed from the points equals the published one."""
    try:
        published = read_passport_data(directory)
        name = str((published.get("model") or {}).get("experiment_id", directory.name))
        trials = measurement_trials(description, records, executor=executor)
        recomputed = build_passport_data(
            description, records, trials, name, assignment, executor=executor
        )
    except IECError as error:
        return CheckResult(PASSPORT_CHECK, False, (f"[{error.error_code}] {error.message}",))
    # The published passport went through JSON: tuples became lists.
    return _listed(PASSPORT_CHECK, compare_passports(published, json.loads(json.dumps(recomputed))))


def compare_passports(published: Any, recomputed: Any) -> list[str]:
    """Return the differences of two passports beyond the tolerance of computation.

    The composition is compared exactly: keys, list lengths, strings, booleans, ``null``,
    integers. Floats under the keys ``fit`` and ``weighting`` - within ``FIRST_STAGE_RTOL``, all the
    others - within ``SECOND_STAGE_RTOL``; every float also has the absolute floor ``ABSOLUTE_FLOOR``.
    Of the records of the candidate forms of a law the chosen one is compared in full, the others by
    the name of the form. The number of steps of a fit is not compared when both passports give one.
    """
    differences: list[str] = []
    _compare(published, recomputed, "", False, differences)
    return differences


def _is_number(value: Any) -> bool:
    """Return whether a value is a number and not a boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(
    published: Any, recomputed: Any, path: str, first_stage: bool, differences: list[str]
) -> None:
    """Compare two values of passports and add every difference to ``differences``."""
    if isinstance(published, Mapping):
        if not isinstance(recomputed, Mapping) or set(published) != set(recomputed):
            differences.append(f"{path or '.'}: the keys differ")
            return
        for key in sorted(published):
            inside = first_stage or key in _FIRST_STAGE_KEYS
            if key == _CANDIDATES and _LAW_FORM in published:
                _compare_candidates(published, recomputed, f"{path}.{key}", inside, differences)
                continue
            if (
                key == _STEPS
                and inside
                and _is_number(published[key])
                and _is_number(recomputed[key])
            ):
                continue
            _compare(published[key], recomputed[key], f"{path}.{key}", inside, differences)
        return
    if isinstance(published, list):
        if not isinstance(recomputed, list) or len(published) != len(recomputed):
            differences.append(f"{path}: the lengths differ")
            return
        for index, (left, right) in enumerate(zip(published, recomputed, strict=True)):
            _compare(left, right, f"{path}[{index}]", first_stage, differences)
        return
    if (
        _is_number(published)
        and _is_number(recomputed)
        and (isinstance(published, float) or isinstance(recomputed, float))
    ):
        if not _floats_agree(float(published), float(recomputed), first_stage):
            differences.append(f"{path}: {published!r} vs {recomputed!r}")
        return
    if type(published) is not type(recomputed) or published != recomputed:
        differences.append(f"{path}: {published!r} vs {recomputed!r}")


def _compare_candidates(
    published_law: Mapping[str, Any],
    recomputed_law: Mapping[str, Any],
    path: str,
    first_stage: bool,
    differences: list[str],
) -> None:
    """Compare the records of the candidate forms of a law: the chosen one in full, the others by name."""
    published, recomputed = published_law[_CANDIDATES], recomputed_law[_CANDIDATES]
    if not isinstance(recomputed, list) or len(published) != len(recomputed):
        differences.append(f"{path}: the lengths differ")
        return
    chosen = published_law[_LAW_FORM]
    for index, (left, right) in enumerate(zip(published, recomputed, strict=True)):
        place = f"{path}[{index}]"
        if isinstance(left, Mapping) and isinstance(right, Mapping) and left.get("form") != chosen:
            _compare(left.get("form"), right.get("form"), f"{place}.form", first_stage, differences)
        else:
            _compare(left, right, place, first_stage, differences)


def _floats_agree(x: float, y: float, first_stage: bool) -> bool:
    """Return whether two floats agree within the tolerance of their stage."""
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    tolerance = FIRST_STAGE_RTOL if first_stage else SECOND_STAGE_RTOL
    return abs(x - y) <= tolerance * max(abs(x), abs(y)) + ABSOLUTE_FLOOR
