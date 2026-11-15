"""Sink that prints every event as one JSON line to a text stream of the console."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from iec.infrastructure.telemetry.sinks.sink_contract import OneTimeWarning
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

_MODULE_NAME = "console_sink"


def standard_error() -> TextIO:
    """Return the current standard error stream of the process."""
    return sys.stderr


def standard_output() -> TextIO:
    """Return the current standard output stream of the process."""
    return sys.stdout


class ConsoleSink:
    """Prints events to a console stream, looked up at every event.

    Looking the stream up at every event follows a replaced ``sys.stderr``: a test capture or a
    redirect set after the sink was built.
    """

    def __init__(self, stream: Callable[[], TextIO] = standard_error) -> None:
        """Remember how to get the stream.

        Args:
            stream: Function returning the stream to print to; standard error by default.
        """
        self._stream = stream
        self._warning = OneTimeWarning(_MODULE_NAME)

    def __call__(self, event: StepTelemetryEvent, /) -> None:
        """Print the event as one JSON line and flush the stream.

        Side effects:
            Writes to the stream; on a write error logs one warning and drops the event.
        """
        stream = self._stream()
        try:
            stream.write(event.as_json_line() + "\n")
            stream.flush()
        except OSError as error:
            self._warning.issue(
                f"telemetry console sink cannot write: {error}; events are dropped"
            )
