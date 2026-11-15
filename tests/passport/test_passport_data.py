"""The passport: plain numbers and the same calculation with a pool of processes and without it."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from iec.contracts import MeasurementDescription, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.measurement_files import read_description, read_points
from iec.passport.passport_data import (
    _plain,
    build_passport_data,
    calculation_pool,
    measurement_trials,
)

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT = ROOT / "artefacts" / "pilot_gpt4o_mini_v55"
ASSIGNMENT = build_law_assignment(load_law_assignment_config(ROOT / "config" / "passport_laws.yaml"))
#: Every tenth answered point of v55: 519 points, on which all three sections have their laws.
STRIDE = 10


def test_plain_numbers_leave_no_numpy_types() -> None:
    value = _plain(
        {"a": np.float64(1.5), "b": [np.int64(2), np.array([1.0, 2.0])], "c": (np.bool_(True),)}
    )
    assert value == {"a": 1.5, "b": [2, [1.0, 2.0]], "c": [True]}
    assert json.dumps(value)


@pytest.fixture(scope="module")
def sample() -> tuple[MeasurementDescription, list[PointRecord]]:
    records = [record for record in read_points(MEASUREMENT) if record.status is PointStatus.OK]
    return read_description(MEASUREMENT), records[::STRIDE]


def test_the_pool_gives_the_same_trials_and_passport(
    sample: tuple[MeasurementDescription, list[PointRecord]],
) -> None:
    description, records = sample
    trials = measurement_trials(description, records)
    passport = build_passport_data(description, records, trials, "sample", ASSIGNMENT)
    with calculation_pool(2) as pool:
        pooled_trials = measurement_trials(description, records, executor=pool)
        pooled = build_passport_data(
            description, records, pooled_trials, "sample", ASSIGNMENT, executor=pool
        )
    assert pooled_trials.points == trials.points
    assert pooled_trials.background == trials.background
    assert json.dumps(pooled, sort_keys=True) == json.dumps(passport, sort_keys=True)


def test_an_error_of_a_worker_keeps_its_code(
    sample: tuple[MeasurementDescription, list[PointRecord]],
) -> None:
    description, records = sample
    broken = [replace(records[0], answer=None), *records[1:3]]
    with calculation_pool(2) as pool, pytest.raises(ContractViolationError) as caught:
        measurement_trials(description, broken, executor=pool)
    assert caught.value.error_code == "POINT_NOT_OK"


def test_one_worker_means_no_pool() -> None:
    with calculation_pool(1) as pool:
        assert pool is None
