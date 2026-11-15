"""The interface of a telemetry sink and the rule for its write failures.

A sink is where a recorded event goes: a file, the console, an external collector. Any callable
that takes an event is a sink, so a sink is a class with ``__call__`` or a plain function. A sink
handles its own write errors: it drops the event and warns once, because telemetry must never
fail the measured operation (specification §8).
"""

from __future__ import annotations

from typing import Protocol

from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent


class TelemetrySink(Protocol):
    """Receiver of recorded telemetry events."""

    def __call__(self, event: StepTelemetryEvent, /) -> None:
        """Take one event; a write error is handled inside and never raised."""


class OneTimeWarning:
    """A warning a sink logs at its first write failure only, so a broken disk cannot flood."""

    def __init__(self, module_name: str) -> None:
        """Remember the module whose name the warning carries."""
        self._module_name = module_name
        self._issued = False

    def issue(self, message: str) -> None:
        """Log the warning on the first call; later calls are silent.

        Side effects:
            Writes one ``WARNING`` record of the package logger on the first call.
        """
        if self._issued:
            return
        self._issued = True
        get_logger(LogContext(module_name=self._module_name)).warning(message)
