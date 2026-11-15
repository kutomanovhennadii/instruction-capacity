"""Mirror tests of the JSONL file sink."""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from iec.infrastructure.telemetry.sinks import JsonlFileSink
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

EventFactory = Callable[..., StepTelemetryEvent]


def test_sink_appends_one_line_per_event(tmp_path: Path, make_event: EventFactory) -> None:
    path = tmp_path / "telemetry.jsonl"
    sink = JsonlFileSink(path)

    sink(make_event(operation="first"))
    sink(make_event(operation="second"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["operation"] for line in lines] == ["first", "second"]


def test_sink_creates_the_directory_of_its_file(tmp_path: Path) -> None:
    sink = JsonlFileSink(tmp_path / "nested" / "telemetry.jsonl")

    assert sink.path.parent.is_dir()


def test_write_error_drops_events_with_one_warning(
    tmp_path: Path,
    make_event: EventFactory,
    isolated_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A folder in place of the file makes every write fail; the measured run goes on."""
    path = tmp_path / "telemetry.jsonl"
    path.mkdir()
    sink = JsonlFileSink(path)

    sink(make_event())
    sink(make_event())

    warnings = [line for line in capsys.readouterr().err.splitlines() if "cannot write" in line]
    assert len(warnings) == 1
