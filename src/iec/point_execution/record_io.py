"""Reading and appending published point records as JSON lines."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from iec.contracts import PointRecord


def append_record(path: Path, record: PointRecord) -> None:
    """Append one record as one JSON line; the parent directory must exist.

    Side effects:
        Creates the file if it does not exist and appends one line to it.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[PointRecord]:
    """Read all records of a JSON-lines file in file order; ``.gz`` files are unpacked.

    Raises:
        ContractViolationError: ``POINT_RECORD_INVALID`` - a line is not a point record.
    """
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
    return [PointRecord.from_dict(json.loads(line)) for line in lines if line.strip()]
