"""verify: the tolerance of the passport comparison and the order of the four checks."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from iec.contracts import (
    GenParams,
    MeasurementDescription,
    ModelConfig,
    PointConditions,
    PointRecord,
    PointStatus,
)
from iec.model_client import MockTokenizer
from iec.passport import verify
from iec.passport.measurement_files import (
    MEASUREMENT_FILENAME,
    PASSPORT_DATA_FILENAME,
    POINTS_FILENAME,
    sha256_of,
)
from iec.passport.verify import compare_passports, verify_measurement
from iec.prompt_generator import RuleGenerator, TextGenerator
from iec.vocab import VocabStore

_VOCAB_DIR = Path(__file__).resolve().parents[2] / "assets" / "vocab"
_CONDITIONS = (
    PointConditions(D=8, n=5, m=2, seed_rules=11, seed_text=12, seed_order=0),
    PointConditions(D=4, n=5, m=1, seed_rules=21, seed_text=22, seed_order=0),
)


def test_identical_passports_have_no_differences() -> None:
    passport = {
        "fit": {"theta": {"x": 1.5}},
        "passport": {"s": {"approximation": {"law": {"a": 2.0, "se": {"a": 0.5}}}}},
    }
    assert compare_passports(passport, json.loads(json.dumps(passport))) == []


def test_the_first_stage_tolerates_rounding_only() -> None:
    assert compare_passports({"theta": 1.0}, {"theta": 1.0 + 1e-9}) == []
    assert compare_passports({"theta": 1.0}, {"theta": 1.0 + 1e-5}) != []


def test_a_second_stage_value_may_move_by_a_hundredth_of_its_se() -> None:
    published = {"approximation": {"row": {"value": 10.0, "se": 2.0}}}
    assert compare_passports(published, {"approximation": {"row": {"value": 10.015, "se": 2.0}}}) == []
    assert compare_passports(published, {"approximation": {"row": {"value": 10.03, "se": 2.0}}}) != []


def test_a_law_parameter_uses_its_se_from_the_se_dictionary() -> None:
    published = {"approximation": {"law": {"a": -2.0, "se": {"a": 0.1}}}}
    moved = {"approximation": {"law": {"a": -2.0009, "se": {"a": 0.1}}}}
    assert compare_passports(published, moved) == []
    moved["approximation"]["law"]["a"] = -2.002
    assert compare_passports(published, moved) != []


def test_other_second_stage_floats_tolerate_ten_percent() -> None:
    published = {"approximation": {"law": {"chi2": 10.0}}}
    assert compare_passports(published, {"approximation": {"law": {"chi2": 10.9}}}) == []
    assert compare_passports(published, {"approximation": {"law": {"chi2": 12.0}}}) != []


def test_the_composition_is_compared_exactly() -> None:
    assert compare_passports({"status": "measured"}, {"status": "extrapolated"}) != []
    assert compare_passports({"n": 3}, {"n": 4}) != []
    assert compare_passports({"value": None}, {"value": 0.0}) != []
    assert compare_passports({"a": 1}, {"a": 1, "b": 2}) != []
    assert compare_passports({"rows": [1.0]}, {"rows": [1.0, 2.0]}) != []


def _measurement(directory: Path, *, tamper_rules: bool = False) -> dict[str, Any]:
    """Write a two-point measurement with a stand-in passport; return the passport."""
    vocab = VocabStore(_VOCAB_DIR).load("v1_intersection")
    records = []
    for index, conditions in enumerate(_CONDITIONS):
        table = RuleGenerator(vocab).generate(conditions)
        text = TextGenerator(vocab, MockTokenizer()).generate(conditions, table)
        rules = tuple((rule.source, rule.target) for rule in table.rules)
        if tamper_rules and index == 0:
            rules = tuple(reversed(rules))
        records.append(
            PointRecord(
                conditions=conditions,
                rules=rules,
                source_text=text.text,
                answer="",
                status=PointStatus.OK,
                prompt_tokens=None,
                response_tokens=None,
            )
        )
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / POINTS_FILENAME, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict()) + "\n")
    description = MeasurementDescription(
        model=ModelConfig(
            model_id="mock-model",
            provider="mock",
            gen_params=GenParams(temperature=0.0, max_tokens=1000, top_p=1.0, seed=None),
        ),
        protocol_version="0.1",
        vocab_id=vocab.vocab_id,
        vocab_hash=vocab.hash,
        files={POINTS_FILENAME: sha256_of(directory / POINTS_FILENAME)},
    )
    (directory / MEASUREMENT_FILENAME).write_text(
        json.dumps(description.as_dict()), encoding="utf-8"
    )
    passport = {"model": {"id": "mock-model", "experiment_id": directory.name}, "fit": {"x": 1.0}}
    (directory / PASSPORT_DATA_FILENAME).write_text(json.dumps(passport), encoding="utf-8")
    return passport


def test_a_consistent_measurement_passes_the_four_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passport = _measurement(tmp_path / "m")
    monkeypatch.setattr(verify, "build_passport_data", lambda description, records, name: passport)
    report = verify_measurement(tmp_path / "m", _VOCAB_DIR)
    assert [check.name for check in report.checks] == ["files", "vocabulary", "generator", "passport"]
    assert report.passed, report.checks


def test_a_changed_points_file_stops_the_other_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passport = _measurement(tmp_path / "m")
    monkeypatch.setattr(verify, "build_passport_data", lambda description, records, name: passport)
    with gzip.open(tmp_path / "m" / POINTS_FILENAME, "at", encoding="utf-8") as handle:
        handle.write("\n")
    report = verify_measurement(tmp_path / "m", _VOCAB_DIR)
    assert not report.passed
    assert report.checks[0].passed is False
    assert all("not run" in check.details[0] for check in report.checks[1:])


def test_a_rule_list_that_the_seeds_do_not_give_fails_the_generator_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passport = _measurement(tmp_path / "m", tamper_rules=True)
    monkeypatch.setattr(verify, "build_passport_data", lambda description, records, name: passport)
    report = verify_measurement(tmp_path / "m", _VOCAB_DIR)
    assert [check.passed for check in report.checks] == [True, True, False, True]


def test_a_different_passport_fails_the_passport_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    passport = _measurement(tmp_path / "m")
    monkeypatch.setattr(
        verify,
        "build_passport_data",
        lambda description, records, name: {**passport, "fit": {"x": 2.0}},
    )
    report = verify_measurement(tmp_path / "m", _VOCAB_DIR)
    assert [check.passed for check in report.checks] == [True, True, True, False]
    assert ".fit.x" in report.checks[3].details[0]
