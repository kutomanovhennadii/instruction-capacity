"""Reading and appending the published records of points as JSON lines.

A file of records holds one JSON object per line (the JSON Lines format): the line end ``\\n``
separates the records, while the other characters that Python counts as line ends, such as
``\\u2028`` or ``\\x85``, may stand inside the answer of a model. Records are only appended, each by
the reliable appending of the infrastructure.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from iec.contracts import PointRecord
from iec.contracts.measurement.point_record import POINT_RECORD_INVALID
from iec.infrastructure.exceptions import ContractViolationError
from iec.infrastructure.files import append_line

#: Suffix of a file of records packed with gzip, as the published measurements are.
PACKED_SUFFIX = ".gz"
#: The separator of the records in a file of records.
RECORD_SEPARATOR = "\n"


def append_record(path: Path, record: PointRecord) -> None:
    """Append one record to a file of records reliably, as one JSON line.

    Args:
        path: The file of records; it is created when missing, and its folder must exist.
        record: The record of a point.

    Raises:
        OSError: the file cannot be read or written.

    Side effects:
        Appends one line by ``append_line``: the whole line after a complete line, synced to the
        disk; a line cut short by a broken-off write is ended first.
    """
    append_line(path, json.dumps(record.as_dict(), ensure_ascii=False))


def read_records(path: Path) -> list[PointRecord]:
    """Read all records of a file of records in file order; a ``.gz`` file is unpacked.

    Args:
        path: The file of records, plain or packed with gzip.

    Returns:
        The records in the order of their lines; empty lines are skipped.

    Raises:
        ContractViolationError: ``POINT_RECORD_INVALID`` - a line is not JSON, such as a record
            cut short by a broken-off write (the message names the file and the line), or the
            JSON object is not a point record; ``FORMAT_VERSION_UNSUPPORTED`` - a record of a
            newer format.
        OSError: the file cannot be read.
    """
    if path.suffix == PACKED_SUFFIX:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8")
    return [
        _parse_record(line, path, number)
        for number, line in enumerate(text.split(RECORD_SEPARATOR), start=1)
        if line.strip()
    ]


def _parse_record(line: str, path: Path, number: int) -> PointRecord:
    """Parse one line of a file of records.

    Args:
        line: The line without its line end.
        path: The file of records, for the message of an error.
        number: The number of the line in the file, from one.

    Returns:
        The point record of the line.

    Raises:
        ContractViolationError: ``POINT_RECORD_INVALID`` - the line is not JSON or the JSON
            object is not a point record.
    """
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as cause:
        raise ContractViolationError.from_cause(
            f"line {number} of {path} is not a JSON record: {cause}",
            cause,
            error_code=POINT_RECORD_INVALID,
        ) from cause
    return PointRecord.from_dict(raw)
