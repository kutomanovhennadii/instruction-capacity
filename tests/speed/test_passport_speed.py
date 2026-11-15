"""The speed of the long operations of the passport against their reference times (specification §10).

The tests time the alignment of the published measurement v55 and the sections of its three
quantities in a pool of the processes of the machine. A test fails when the operation takes more than
``SLACK`` times its reference time from ``reference_times.json``, so an acceleration cannot roll back
unnoticed. The tests are slow and depend on the machine: they run only on ``-m speed``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

import pytest

from iec.passport.laws.forms.law_assignment import build_law_assignment, load_law_assignment_config
from iec.passport.measurement_files import read_description, read_points
from iec.passport.passport_data import (
    MeasurementTrials,
    build_passport_data,
    calculation_pool,
    measurement_trials,
)

ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT = ROOT / "artefacts" / "pilot_gpt4o_mini_v55"
#: How many times its reference time an operation may take before the test fails.
SLACK = 1.5

pytestmark = pytest.mark.speed


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        Path(__file__).with_name("reference_times.json").read_text(encoding="utf-8")
    )
    return loaded


@pytest.fixture(scope="module")
def pool() -> Iterator[Executor | None]:
    with calculation_pool(os.cpu_count() or 1) as executor:
        yield executor


@pytest.fixture(scope="module")
def timed_trials(pool: Executor | None) -> tuple[MeasurementTrials, float]:
    description, records = read_description(MEASUREMENT), read_points(MEASUREMENT)
    started = time.perf_counter()
    trials = measurement_trials(description, records, executor=pool)
    return trials, time.perf_counter() - started


def test_the_alignment_keeps_its_speed(
    timed_trials: tuple[MeasurementTrials, float], reference: dict[str, Any]
) -> None:
    _, seconds = timed_trials
    assert seconds <= SLACK * reference["alignment_seconds"]


def test_the_sections_keep_their_speed(
    pool: Executor | None,
    timed_trials: tuple[MeasurementTrials, float],
    reference: dict[str, Any],
) -> None:
    description, records = read_description(MEASUREMENT), read_points(MEASUREMENT)
    assignment = build_law_assignment(
        load_law_assignment_config(ROOT / "config" / "passport_laws.yaml")
    )
    trials, _ = timed_trials
    started = time.perf_counter()
    build_passport_data(description, records, trials, MEASUREMENT.name, assignment, executor=pool)
    assert time.perf_counter() - started <= SLACK * reference["sections_seconds"]
