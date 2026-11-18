"""The saved fit of a passport: the passport from the file and its format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iec.contracts import PointStatus
from iec.infrastructure.exceptions import ContractViolationError, PassportError
from iec.passport.fit_store import (
    FITS_FORMAT_VERSION,
    PASSPORT_FITS_INVALID,
    PassportFits,
    read_passport_fits,
    write_passport_fits,
)
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.measurement_files import read_description, read_points
from iec.passport.passport_data import measurement_trials, passport_fits, passport_from_fits

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT = ROOT / "artefacts" / "pilot_gpt4o_mini_v55"
ASSIGNMENT = build_law_assignment(load_law_assignment_config(ROOT / "config" / "passport_laws.yaml"))
#: Every tenth answered point of v55: all three sections have their laws on them.
STRIDE = 10


@pytest.fixture(scope="module")
def fits() -> PassportFits:
    records = [record for record in read_points(MEASUREMENT) if record.status is PointStatus.OK]
    sample = records[::STRIDE]
    description = read_description(MEASUREMENT)
    return passport_fits(description, sample, measurement_trials(description, sample), "sample")


def test_the_passport_from_the_saved_fit_is_the_passport_of_its_fits(
    fits: PassportFits, tmp_path: Path
) -> None:
    restored = read_passport_fits(write_passport_fits(fits, tmp_path / "fits.json"))
    assert json.dumps(passport_from_fits(restored, ASSIGNMENT), sort_keys=True) == json.dumps(
        passport_from_fits(fits, ASSIGNMENT), sort_keys=True
    )


def test_the_saved_fit_has_three_quantities_and_no_rows_of_a_fit(
    fits: PassportFits, tmp_path: Path
) -> None:
    data = json.loads(write_passport_fits(fits, tmp_path / "fits.json").read_text(encoding="utf-8"))
    assert data["format_version"] == FITS_FORMAT_VERSION
    assert list(data["sections"]) == ["P_sync", "H", "F"]
    assert data["sections"]["F"]["kind"] == "answer_share"
    assert "x" not in data["sections"]["P_sync"]["design"]


def test_a_missing_file_and_a_newer_format_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PassportError) as missing:
        read_passport_fits(tmp_path / "absent.json")
    assert missing.value.error_code == PASSPORT_FITS_INVALID
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"format_version": FITS_FORMAT_VERSION + 1}), encoding="utf-8")
    with pytest.raises(ContractViolationError) as caught:
        read_passport_fits(newer)
    assert caught.value.error_code == "FORMAT_VERSION_UNSUPPORTED"
