"""Mirror tests of the reading and the reliable appending of point records."""

import dataclasses
import gzip
import json
from pathlib import Path

import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError
from iec.point_execution import append_record, read_records

_RECORD = PointRecord(
    conditions=PointConditions(D=2, n=1, m=1, seed_rules=1, seed_text=2, seed_order=0),
    rules=(("lake", "stone"),),
    source_text="lake tree",
    answer="stone tree",
    status=PointStatus.OK,
    prompt_tokens=10,
    response_tokens=2,
)


def test_an_answer_with_unicode_line_separators_reads_back(tmp_path: Path) -> None:
    # U+2028 and U+0085 stay raw in JSON and are line ends for str.splitlines, not for JSON Lines.
    record = dataclasses.replace(_RECORD, answer="stone\u2028tree\x85")
    path = tmp_path / "points.jsonl"

    append_record(path, record)

    assert read_records(path) == [record]


def test_a_packed_file_is_read_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "points.jsonl.gz"
    second = dataclasses.replace(_RECORD, status=PointStatus.NA, answer=None)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in (_RECORD, second):
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")

    assert read_records(path) == [_RECORD, second]


def test_a_cut_record_is_named_and_the_next_record_stays_whole(tmp_path: Path) -> None:
    path = tmp_path / "points.jsonl"
    append_record(path, _RECORD)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"format_version": 1, "D": 2')  # a write broken off in the middle
    append_record(path, _RECORD)

    with pytest.raises(ContractViolationError) as caught:
        read_records(path)

    assert caught.value.error_code == "POINT_RECORD_INVALID"
    assert "line 2" in str(caught.value)
    lines = path.read_text(encoding="utf-8").split("\n")
    assert PointRecord.from_dict(json.loads(lines[2])) == _RECORD
