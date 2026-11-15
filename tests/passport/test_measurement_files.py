"""The files of a measurement and of its passport: the points in one file or in numbered parts, the
sorted keys and the format version."""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.contracts.format_version import FORMAT_VERSION_UNSUPPORTED
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport.measurement_files import (
    PASSPORT_DATA_FILENAME,
    PASSPORT_DATA_FORMAT_VERSION,
    POINTS_FILENAME,
    points_names,
    read_passport_data,
    read_points,
    write_passport_data,
)

_RECORD = PointRecord(
    conditions=PointConditions(D=2, n=1, m=1, seed_rules=1, seed_text=2, seed_order=0),
    rules=(("lake", "stone"),),
    source_text="lake tree",
    answer="stone tree",
    status=PointStatus.OK,
    prompt_tokens=10,
    response_tokens=2,
)


def _pack(path: Path, records: list[PointRecord]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def test_the_points_are_read_from_numbered_parts_in_the_order_of_their_numbers(tmp_path: Path) -> None:
    second = dataclasses.replace(_RECORD, status=PointStatus.NA, answer=None)
    _pack(tmp_path / "points-002.jsonl.gz", [second])
    _pack(tmp_path / "points-001.jsonl.gz", [_RECORD])
    assert points_names(tmp_path) == ["points-001.jsonl.gz", "points-002.jsonl.gz"]
    assert read_points(tmp_path) == [_RECORD, second]


def test_the_single_points_file_is_read_when_it_exists(tmp_path: Path) -> None:
    _pack(tmp_path / POINTS_FILENAME, [_RECORD])
    assert points_names(tmp_path) == [POINTS_FILENAME]
    assert read_points(tmp_path) == [_RECORD]


def test_without_points_the_single_file_is_named_and_reported_missing(tmp_path: Path) -> None:
    assert points_names(tmp_path) == [POINTS_FILENAME]
    with pytest.raises(PassportError) as caught:
        read_points(tmp_path)
    assert caught.value.error_code == "MEASUREMENT_FILES_MISSING"


def test_the_file_has_sorted_keys(tmp_path: Path) -> None:
    path = write_passport_data({"b": 1, "a": 2}, tmp_path)
    assert path.name == PASSPORT_DATA_FILENAME
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index(
        '"b"'
    )


def test_the_file_carries_its_format_version_and_reads_back_without_it(tmp_path: Path) -> None:
    path = write_passport_data({"points": 3}, tmp_path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {"format_version": PASSPORT_DATA_FORMAT_VERSION, "points": 3}
    assert read_passport_data(tmp_path) == {"points": 3}


def test_a_file_without_the_version_is_of_version_one(tmp_path: Path) -> None:
    (tmp_path / PASSPORT_DATA_FILENAME).write_text(json.dumps({"points": 3}), encoding="utf-8")
    assert read_passport_data(tmp_path) == {"points": 3}


def test_a_newer_format_is_refused(tmp_path: Path) -> None:
    (tmp_path / PASSPORT_DATA_FILENAME).write_text(
        json.dumps({"format_version": PASSPORT_DATA_FORMAT_VERSION + 1}), encoding="utf-8"
    )
    with pytest.raises(ContractViolationError) as caught:
        read_passport_data(tmp_path)
    assert caught.value.error_code == FORMAT_VERSION_UNSUPPORTED
