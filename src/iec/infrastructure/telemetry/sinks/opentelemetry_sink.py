"""Sink that turns every event into an OpenTelemetry span for external collectors.

OpenTelemetry is the open standard of telemetry collection: through it the events reach
Prometheus, Grafana and other external collectors, as in a microservice (specification §8). It is
an optional dependency, so this module does not import it: the sink works with any tracer that
has the method it calls, and the sink registry builds a real tracer only when the configuration
asks for this sink. The span keeps the times of the call, the fields of the event as attributes
under the prefix ``iec.`` and the error status of a failed call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent, TelemetryStatus

#: Value of a span attribute: OpenTelemetry takes strings, integers, floats and booleans.
AttributeValue = str | int | float | bool
#: Function that makes the error status of a span from the error message.
ErrorStatusFactory = Callable[[str], object]
#: Prefix of the attributes of an event in a span.
ATTRIBUTE_PREFIX = "iec."
_NANOSECONDS_PER_SECOND = 1_000_000_000


class TelemetrySpan(Protocol):
    """The part of an OpenTelemetry span the sink uses."""

    def set_status(self, status: object, /) -> None:
        """Set the status of the span."""

    def end(self, end_time: int | None = None) -> None:
        """End the span at the given time in nanoseconds since the epoch."""


class SpanTracer(Protocol):
    """The part of an OpenTelemetry tracer the sink uses."""

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
        start_time: int | None = None,
    ) -> TelemetrySpan:
        """Start a span with attributes at the given time in nanoseconds since the epoch."""
        ...


class OpenTelemetrySink:
    """Turns events into finished spans of a tracer."""

    def __init__(self, tracer: SpanTracer, error_status: ErrorStatusFactory) -> None:
        """Remember the tracer and the maker of the error status.

        Args:
            tracer: OpenTelemetry tracer or any object with the same ``start_span``.
            error_status: Function making the error status of a span from an error message.
        """
        self._tracer = tracer
        self._error_status = error_status

    def __call__(self, event: StepTelemetryEvent, /) -> None:
        """Record the event as one finished span named ``<block>.<operation>``."""
        span = self._tracer.start_span(
            f"{event.block}.{event.operation}",
            attributes=span_attributes(event),
            start_time=_nanoseconds(event.started_at),
        )
        if event.status is TelemetryStatus.FAILURE:
            message = event.error_message or event.error_type or "failure"
            span.set_status(self._error_status(message))
        span.end(end_time=_nanoseconds(event.ended_at))


def span_attributes(event: StepTelemetryEvent) -> dict[str, AttributeValue]:
    """Return the attributes of the span of an event: every set field under ``iec.``.

    A span attribute cannot be ``None`` or a mapping, so unset fields are left out, and a mapping
    field such as ``usage`` becomes one attribute per key: ``iec.usage.prompt_tokens``.
    """
    attributes: dict[str, AttributeValue] = {}
    for name, value in event.as_dict().items():
        if isinstance(value, Mapping):
            attributes.update(
                {f"{ATTRIBUTE_PREFIX}{name}.{key}": item for key, item in value.items()}
            )
        elif isinstance(value, str | int | float | bool):
            attributes[f"{ATTRIBUTE_PREFIX}{name}"] = value
    return attributes


def _nanoseconds(seconds: float) -> int:
    """Return seconds since the epoch as integer nanoseconds, the time unit of OpenTelemetry."""
    return int(seconds * _NANOSECONDS_PER_SECOND)
