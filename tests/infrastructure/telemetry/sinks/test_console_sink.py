"""Mirror tests of the console sink."""

import io
import json
import logging
from collections.abc import Callable

import pytest

from iec.infrastructure.telemetry.sinks import ConsoleSink
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

EventFactory = Callable[..., StepTelemetryEvent]


class BrokenStream(io.StringIO):
    """A stream whose every write fails, as a closed console does."""

    def write(self, text: str) -> int:
        raise OSError("stream closed")


def test_sink_prints_one_json_line_to_its_stream(make_event: EventFactory) -> None:
    buffer = io.StringIO()
    sink = ConsoleSink(lambda: buffer)

    sink(make_event(operation="model_generate"))

    assert json.loads(buffer.getvalue())["operation"] == "model_generate"
    assert buffer.getvalue().endswith("\n")


def test_sink_prints_to_standard_error_by_default(
    make_event: EventFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    ConsoleSink()(make_event())

    assert '"operation": "model_generate"' in capsys.readouterr().err


def test_write_error_drops_events_with_one_warning(
    make_event: EventFactory,
    isolated_package_logger: logging.Logger,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sink = ConsoleSink(BrokenStream)

    sink(make_event())
    sink(make_event())

    warnings = [line for line in capsys.readouterr().err.splitlines() if "cannot write" in line]
    assert len(warnings) == 1
