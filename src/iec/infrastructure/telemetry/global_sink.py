"""Process-wide telemetry sink: attaching a file without rebinding decorators."""

from __future__ import annotations

from collections.abc import Callable

from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

TelemetrySink = Callable[[StepTelemetryEvent], None]

_global_sink: TelemetrySink | None = None


def set_telemetry_sink(sink: TelemetrySink) -> None:
    """Set the process-wide sink of telemetry events.

    Args:
        sink: Event handler called after logging.

    Side effects:
        Replaces the previously set sink in the module state of the process.
    """
    global _global_sink
    _global_sink = sink


def reset_telemetry_sink() -> None:
    """Remove the process-wide sink, restoring the default silence.

    Side effects:
        Resets the module state of the process to ``None``.
    """
    global _global_sink
    _global_sink = None


def get_telemetry_sink() -> TelemetrySink | None:
    """Return the current process-wide sink or ``None`` when none is set."""
    return _global_sink
