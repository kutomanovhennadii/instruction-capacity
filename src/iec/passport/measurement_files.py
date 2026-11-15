"""Files of a published measurement directory and of its passport.

A measurement directory holds ``measurement.json`` - the description with the SHA-256 of every data
file - and ``points.jsonl.gz`` - the point records in the published order. Points that would make one
file larger than the 100 MiB limit of a file on GitHub are kept in numbered parts instead,
``points-001.jsonl.gz``, ``points-002.jsonl.gz`` and so on, read one after another in the order of
their numbers; the description lists every part. The passport command adds
``passport_data.json``, ``passport.md`` and ``figures/``. ``passport_data.json`` carries
``format_version``: version 1 is the passport of version 0.2.0, and a file without the field is of
version 1 by the rule of ``contracts.format_version``; a newer version is refused. The passport file
is written whole or not at all: through a temporary file and a rename.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iec.contracts import MeasurementDescription, PointRecord
from iec.contracts.format_version import check_format_version
from iec.infrastructure.exceptions import PassportError
from iec.infrastructure.files import write_text_atomically
from iec.point_execution import read_records

MEASUREMENT_FILENAME = "measurement.json"
POINTS_FILENAME = "points.jsonl.gz"
#: The numbered parts of the points, read in the order of their numbers.
POINTS_PART_GLOB = "points-[0-9][0-9][0-9].jsonl.gz"
PASSPORT_DATA_FILENAME = "passport_data.json"
PASSPORT_FILENAME = "passport.md"
FIGURES_DIRNAME = "figures"
#: The version of the format of ``passport_data.json`` that this code writes.
PASSPORT_DATA_FORMAT_VERSION = 1
#: Error code of a passport data file that is not a passport of a known format.
PASSPORT_DATA_INVALID = "PASSPORT_DATA_INVALID"

_MISSING_CODE = "MEASUREMENT_FILES_MISSING"


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: The file.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_file(path: Path) -> Path:
    """Return the path of a file of the measurement that must exist.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
    """
    if not path.is_file():
        raise PassportError(f"'{path}' does not exist", error_code=_MISSING_CODE)
    return path


def read_description(directory: Path) -> MeasurementDescription:
    """Read ``measurement.json`` of a measurement directory.

    Args:
        directory: The measurement directory.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
    """
    path = _require_file(directory / MEASUREMENT_FILENAME)
    return MeasurementDescription.from_dict(json.loads(path.read_text(encoding="utf-8")))


def points_names(directory: Path) -> list[str]:
    """Return the names of the files of the point records in the order of reading.

    The points are ``points.jsonl.gz`` or, when only they exist, its numbered parts in the order of
    their numbers. When neither exists, the name of the single file is returned, so that a check can
    name the missing file.

    Args:
        directory: The measurement directory.
    """
    parts = sorted(path.name for path in directory.glob(POINTS_PART_GLOB))
    if parts and not (directory / POINTS_FILENAME).is_file():
        return parts
    return [POINTS_FILENAME]


def read_points(directory: Path) -> list[PointRecord]:
    """Read the point records of a measurement directory in file order: ``points.jsonl.gz`` or its
    numbered parts one after another.

    Args:
        directory: The measurement directory.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the points file does not exist.
    """
    return [
        record
        for name in points_names(directory)
        for record in read_records(_require_file(directory / name))
    ]


def write_passport_data(passport: dict[str, Any], directory: Path) -> Path:
    """Write ``passport_data.json`` with its format version, reliably.

    The keys are sorted and the indent is fixed: the same passport gives the same bytes.

    Args:
        passport: The dictionary of the passport (``passport_data.build_passport_data``).
        directory: The folder of the passport.

    Returns:
        The path of the file.

    Side effects:
        Creates or replaces the file through a temporary file and a rename.
    """
    path = directory / PASSPORT_DATA_FILENAME
    stored = {"format_version": PASSPORT_DATA_FORMAT_VERSION, **passport}
    write_text_atomically(
        path, json.dumps(stored, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return path


def read_passport_data(directory: Path) -> dict[str, Any]:
    """Read ``passport_data.json`` of a directory: the passport without its format version.

    Args:
        directory: The folder of the passport.

    Returns:
        The dictionary of the passport as ``passport_data.build_passport_data`` gives it.

    Raises:
        PassportError: ``MEASUREMENT_FILES_MISSING`` - the file does not exist.
        ContractViolationError: ``PASSPORT_DATA_INVALID`` - the file is not a JSON object or its
            version is not a positive integer; ``FORMAT_VERSION_UNSUPPORTED`` - the version is newer
            than this code writes.
    """
    path = _require_file(directory / PASSPORT_DATA_FILENAME)
    data = json.loads(path.read_text(encoding="utf-8"))
    check_format_version(
        data, PASSPORT_DATA_FORMAT_VERSION, invalid_code=PASSPORT_DATA_INVALID, what="the passport"
    )
    return {key: value for key, value in data.items() if key != "format_version"}
