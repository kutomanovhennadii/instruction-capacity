"""Files of a published measurement directory and of its passport.

A measurement directory holds ``measurement.json`` - the description with the SHA-256 of every
data file - and ``points.jsonl.gz`` - the point records in the published order. The passport
command adds ``passport_data.json``, ``passport.md`` and ``figures/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iec.contracts import MeasurementDescription, PointRecord
from iec.infrastructure.exceptions import PassportError
from iec.point_execution import read_records

MEASUREMENT_FILENAME = "measurement.json"
POINTS_FILENAME = "points.jsonl.gz"
PASSPORT_DATA_FILENAME = "passport_data.json"
PASSPORT_FILENAME = "passport.md"
FIGURES_DIRNAME = "figures"

_MISSING_CODE = "MEASUREMENT_FILES_MISSING"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise PassportError(f"'{path}' does not exist", error_code=_MISSING_CODE)
    return path


def read_description(directory: Path) -> MeasurementDescription:
    """Read ``measurement.json`` of a measurement directory.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
    """
    path = _require_file(directory / MEASUREMENT_FILENAME)
    return MeasurementDescription.from_dict(json.loads(path.read_text(encoding="utf-8")))


def read_points(directory: Path) -> list[PointRecord]:
    """Read the point records of a measurement directory in file order.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
    """
    return read_records(_require_file(directory / POINTS_FILENAME))


def write_passport_data(passport: dict[str, Any], directory: Path) -> Path:
    """Write ``passport_data.json``: sorted keys and a fixed indent - the same passport gives
    the same bytes."""
    path = directory / PASSPORT_DATA_FILENAME
    path.write_text(
        json.dumps(passport, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_passport_data(directory: Path) -> dict[str, Any]:
    """Read ``passport_data.json`` of a directory.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
    """
    path = _require_file(directory / PASSPORT_DATA_FILENAME)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
