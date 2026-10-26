"""Verification of a published measurement: integrity and reproducibility.

Four checks, in this order: (1) the SHA-256 of every file listed in ``measurement.json``;
(2) the vocabulary - its hash equals the description, and it matches its reference version;
(3) the generator rebuilds exactly the recorded rule list and source text of every point from
its seeds; (4) the passport recomputed from the points equals the published one - the
composition exactly, the numbers within the tolerance of computation. ``verify`` checks
integrity and reproducibility only; it does not prove the method right.

**Tolerance of check 4.** Reordering the summation of the same points - a perturbation at the
level of rounding, the same a different machine makes - moved the published passports of two
measurements by at most 1.5e-9 relative in the first stage (estimates, errors and coefficients
of bins by 3.1e-12, near-zero elements of the covariance by 1.5e-9), by at most 3.5e-4 of
their own SE in the numbers of the second stage that have one, and by at most 3.2e-2 relative
in the other numbers of the second stage (the errors of a degenerate density law, where two
parameters trade off along a flat valley). The tolerances below keep a margin of 3 to 600
times over these spreads.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iec.contracts import MeasurementDescription, PointRecord, Vocab
from iec.infrastructure.exceptions import IECError
from iec.model_client import MockTokenizer
from iec.passport.measurement_files import (
    POINTS_FILENAME,
    read_description,
    read_passport_data,
    read_points,
    sha256_of,
)
from iec.passport.passport_data import build_passport_data
from iec.prompt_generator import RuleGenerator, TextGenerator
from iec.vocab import VocabStore, VocabVerifier

#: Relative tolerance of the floats outside the second stage.
FIRST_STAGE_RTOL = 1e-6
#: A float of the second stage with its own SE may move by this share of the SE.
SECOND_STAGE_SE_SHARE = 0.01
#: Relative tolerance of the other floats of the second stage.
SECOND_STAGE_RTOL = 0.1
#: Absolute floor of every float tolerance: a difference of zeros.
ABSOLUTE_FLOOR = 1e-12

FILES_CHECK = "files"
VOCABULARY_CHECK = "vocabulary"
GENERATOR_CHECK = "generator"
PASSPORT_CHECK = "passport"

#: The block of the second stage inside the passport section.
_SECOND_STAGE_KEY = "approximation"
#: Keys of a number whose own SE is the scalar ``se`` of the same record.
_VALUE_KEYS = ("value", "mu_v")
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


def verify_measurement(directory: Path, vocab_dir: Path) -> VerificationReport:
    """Run the four checks on a published measurement directory.

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
    passport = _check_passport(directory, description, records)
    return VerificationReport((files, vocabulary, generator, passport))


def _listed(name: str, differences: Sequence[str]) -> CheckResult:
    """Return a check result with at most ``_SHOWN`` differences and the count of the rest."""
    shown = tuple(differences[:_SHOWN])
    if len(differences) > _SHOWN:
        shown += (f"... {len(differences)} difference(s) in all",)
    return CheckResult(name, not differences, shown)


def _check_files(directory: Path, description: MeasurementDescription) -> CheckResult:
    """Check 1: every listed file exists and has its SHA-256; the points file is listed."""
    differences: list[str] = []
    if POINTS_FILENAME not in description.files:
        differences.append(f"{POINTS_FILENAME} is not listed in the description")
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
        return CheckResult(VOCABULARY_CHECK, False, (f"[{error.error_code}] {error.message}",)), None
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
    directory: Path, description: MeasurementDescription, records: Sequence[PointRecord]
) -> CheckResult:
    """Check 4: the passport recomputed from the points equals the published one."""
    try:
        published = read_passport_data(directory)
        name = str((published.get("model") or {}).get("experiment_id", directory.name))
        recomputed = build_passport_data(description, records, name)
    except IECError as error:
        return CheckResult(PASSPORT_CHECK, False, (f"[{error.error_code}] {error.message}",))
    # The published passport went through JSON: tuples became lists.
    return _listed(PASSPORT_CHECK, compare_passports(published, json.loads(json.dumps(recomputed))))


def compare_passports(published: Any, recomputed: Any) -> list[str]:
    """Return the differences of two passports beyond the tolerance of computation.

    The composition is compared exactly: keys, list lengths, strings, booleans, ``null``,
    integers. Floats outside the block ``approximation`` - within ``FIRST_STAGE_RTOL``; inside
    it, a number with its own SE in the same record - within ``SECOND_STAGE_SE_SHARE`` of that
    SE, any other float - within ``SECOND_STAGE_RTOL``.
    """
    differences: list[str] = []
    _compare(published, recomputed, "", False, None, differences)
    return differences


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _own_se(record: Mapping[str, Any], key: str) -> float | None:
    """Return the SE standing next to the number ``key`` of a record, if there is one."""
    se: Any = record.get("se")
    if key in _VALUE_KEYS and _is_number(se):
        return float(se)
    if isinstance(se, Mapping) and key != "se" and _is_number(se.get(key)):
        return float(se[key])
    return None


def _compare(
    published: Any,
    recomputed: Any,
    path: str,
    second_stage: bool,
    own_se: float | None,
    differences: list[str],
) -> None:
    if isinstance(published, Mapping):
        if not isinstance(recomputed, Mapping) or set(published) != set(recomputed):
            differences.append(f"{path or '.'}: the keys differ")
            return
        for key in sorted(published):
            inside = second_stage or key == _SECOND_STAGE_KEY
            child_se = _own_se(published, key) if inside else None
            _compare(
                published[key], recomputed[key], f"{path}.{key}", inside, child_se, differences
            )
        return
    if isinstance(published, list):
        if not isinstance(recomputed, list) or len(published) != len(recomputed):
            differences.append(f"{path}: the lengths differ")
            return
        for index, (left, right) in enumerate(zip(published, recomputed, strict=True)):
            _compare(left, right, f"{path}[{index}]", second_stage, None, differences)
        return
    if (
        _is_number(published)
        and _is_number(recomputed)
        and (isinstance(published, float) or isinstance(recomputed, float))
    ):
        if not _floats_agree(float(published), float(recomputed), second_stage, own_se):
            differences.append(f"{path}: {published!r} vs {recomputed!r}")
        return
    if type(published) is not type(recomputed) or published != recomputed:
        differences.append(f"{path}: {published!r} vs {recomputed!r}")


def _floats_agree(x: float, y: float, second_stage: bool, own_se: float | None) -> bool:
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    if second_stage and own_se is not None:
        limit = SECOND_STAGE_SE_SHARE * abs(own_se) + ABSOLUTE_FLOOR
    elif second_stage:
        limit = SECOND_STAGE_RTOL * max(abs(x), abs(y)) + ABSOLUTE_FLOOR
    else:
        limit = FIRST_STAGE_RTOL * max(abs(x), abs(y)) + ABSOLUTE_FLOOR
    return abs(x - y) <= limit
