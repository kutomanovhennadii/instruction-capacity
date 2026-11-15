"""Sinks of telemetry events behind one interface: memory, a JSONL file, console, OpenTelemetry."""

from iec.infrastructure.telemetry.sinks.console_sink import (
    ConsoleSink,
    standard_error,
    standard_output,
)
from iec.infrastructure.telemetry.sinks.jsonl_file_sink import JsonlFileSink
from iec.infrastructure.telemetry.sinks.memory_sink import MemorySink
from iec.infrastructure.telemetry.sinks.opentelemetry_sink import (
    OpenTelemetrySink,
    SpanTracer,
    TelemetrySpan,
    span_attributes,
)
from iec.infrastructure.telemetry.sinks.sink_contract import OneTimeWarning, TelemetrySink

__all__ = [
    "ConsoleSink",
    "JsonlFileSink",
    "MemorySink",
    "OneTimeWarning",
    "OpenTelemetrySink",
    "SpanTracer",
    "TelemetrySink",
    "TelemetrySpan",
    "span_attributes",
    "standard_error",
    "standard_output",
]
