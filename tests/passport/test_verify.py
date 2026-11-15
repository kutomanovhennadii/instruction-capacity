"""The checks of a published measurement and the tolerance of the comparison of passports."""

import gzip
import json
from pathlib import Path

import pytest

from iec.contracts import GenParams, MeasurementDescription, ModelConfig
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.measurement_files import (
    MEASUREMENT_FILENAME,
    POINTS_FILENAME,
    sha256_of,
)
from iec.passport.verify import (
    ABSOLUTE_FLOOR,
    FILES_CHECK,
    FIRST_STAGE_RTOL,
    PASSPORT_CHECK,
    SECOND_STAGE_RTOL,
    compare_passports,
    verify_measurement,
)


#: The forms of the laws of the new folder.
ASSIGNMENT = build_law_assignment(
    load_law_assignment_config(Path(__file__).resolve().parents[2] / "config" / "passport_laws.yaml")
)


def _description(files: dict[str, str]) -> MeasurementDescription:
    return MeasurementDescription(
        model=ModelConfig(model_id="mock-model", provider="mock", gen_params=GenParams()),
        protocol_version="0.2",
        vocab_id="v1_intersection",
        vocab_hash="sha256:none",
        files=files,
    )


def test_equal_passports_have_no_differences() -> None:
    passport = {"fit": {"sigma2": 2.0}, "laws": {"kappa": -0.5, "names": ["a", 1, None, True]}}
    assert compare_passports(passport, json.loads(json.dumps(passport))) == []


def test_the_composition_is_compared_exactly() -> None:
    assert compare_passports({"a": 1}, {"b": 1}) == [".: the keys differ"]
    assert compare_passports({"a": [1, 2]}, {"a": [1]}) == [".a: the lengths differ"]
    assert compare_passports({"a": "x"}, {"a": "y"}) == [".a: 'x' vs 'y'"]
    assert compare_passports({"a": 3}, {"a": 4}) == [".a: 3 vs 4"]
    assert compare_passports({"a": None}, {"a": 1.0}) == [".a: None vs 1.0"]
    assert compare_passports({"a": True}, {"a": 1}) == [".a: True vs 1"]


def test_a_number_of_the_fit_has_the_narrow_tolerance() -> None:
    near = 1.0 + 0.5 * FIRST_STAGE_RTOL
    far = 1.0 + 2.0 * FIRST_STAGE_RTOL
    assert compare_passports({"H": {"fit": {"s": 1.0}}}, {"H": {"fit": {"s": near}}}) == []
    assert compare_passports({"H": {"fit": {"s": 1.0}}}, {"H": {"fit": {"s": far}}}) != []
    assert compare_passports({"weighting": {"n": 1.0}}, {"weighting": {"n": far}}) != []


def test_a_number_computed_from_the_fit_has_the_wide_tolerance() -> None:
    near = 1.0 + 0.5 * SECOND_STAGE_RTOL
    far = 1.0 + 2.0 * SECOND_STAGE_RTOL
    assert compare_passports({"H": {"laws": {"k": [1.0]}}}, {"H": {"laws": {"k": [near]}}}) == []
    assert compare_passports({"H": {"laws": {"k": [1.0]}}}, {"H": {"laws": {"k": [far]}}}) != []


def test_a_number_near_zero_is_compared_by_the_floor() -> None:
    assert compare_passports({"thresholds": {"q": 1e-16}}, {"thresholds": {"q": -1e-16}}) == []
    assert compare_passports({"thresholds": {"q": 0.0}}, {"thresholds": {"q": 2 * ABSOLUTE_FLOOR}}) != []


def test_a_file_with_another_checksum_stops_the_checks(tmp_path: Path) -> None:
    (tmp_path / POINTS_FILENAME).write_bytes(b"points")
    description = _description({POINTS_FILENAME: "0" * 64})
    (tmp_path / MEASUREMENT_FILENAME).write_text(json.dumps(description.as_dict()), encoding="utf-8")
    report = verify_measurement(tmp_path, tmp_path, ASSIGNMENT)
    assert not report.passed
    assert report.checks[0].name == FILES_CHECK
    assert "SHA-256 differs" in report.checks[0].details[0]
    assert all(check.details[0].startswith("not run") for check in report.checks[1:])


def test_a_listed_file_that_does_not_exist_is_reported(tmp_path: Path) -> None:
    description = _description({POINTS_FILENAME: "0" * 64})
    (tmp_path / MEASUREMENT_FILENAME).write_text(json.dumps(description.as_dict()), encoding="utf-8")
    report = verify_measurement(tmp_path, tmp_path, ASSIGNMENT)
    assert "the file does not exist" in report.checks[0].details[0]


def test_every_numbered_part_of_the_points_must_be_listed(tmp_path: Path) -> None:
    first, second = tmp_path / "points-001.jsonl.gz", tmp_path / "points-002.jsonl.gz"
    for part in (first, second):
        with gzip.open(part, "wb"):
            pass
    description = _description({first.name: sha256_of(first)})
    (tmp_path / MEASUREMENT_FILENAME).write_text(json.dumps(description.as_dict()), encoding="utf-8")
    report = verify_measurement(tmp_path, tmp_path, ASSIGNMENT)
    assert report.checks[0].details == ("points-002.jsonl.gz is not listed in the description",)


def test_listed_numbered_parts_pass_the_files_check(tmp_path: Path) -> None:
    parts = [tmp_path / "points-001.jsonl.gz", tmp_path / "points-002.jsonl.gz"]
    for part in parts:
        with gzip.open(part, "wb"):
            pass
    description = _description({part.name: sha256_of(part) for part in parts})
    (tmp_path / MEASUREMENT_FILENAME).write_text(json.dumps(description.as_dict()), encoding="utf-8")
    report = verify_measurement(
        tmp_path, Path(__file__).resolve().parents[2] / "assets" / "vocab", ASSIGNMENT
    )
    assert report.checks[0].name == FILES_CHECK
    assert report.checks[0].passed


def test_a_measurement_without_a_passport_fails_the_passport_check(tmp_path: Path) -> None:
    points = tmp_path / POINTS_FILENAME
    with gzip.open(points, "wb"):
        pass
    description = _description({POINTS_FILENAME: sha256_of(points)})
    (tmp_path / MEASUREMENT_FILENAME).write_text(json.dumps(description.as_dict()), encoding="utf-8")
    report = verify_measurement(
        tmp_path, Path(__file__).resolve().parents[2] / "assets" / "vocab", ASSIGNMENT
    )
    passport = next(check for check in report.checks if check.name == PASSPORT_CHECK)
    assert not passport.passed
    assert "MEASUREMENT_FILES_MISSING" in passport.details[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_special_floats_agree_only_with_themselves(value: float) -> None:
    assert compare_passports({"a": value}, {"a": value}) == []
    assert compare_passports({"a": value}, {"a": 1.0}) != []
