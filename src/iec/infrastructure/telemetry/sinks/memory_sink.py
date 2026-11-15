"""Sink that keeps the events of a run in memory: for tests and for counting inside a run."""

from __future__ import annotations

from dataclasses import dataclass, field

from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent


@dataclass(slots=True)
class MemorySink:
    """Collector of events in the order they come."""

    events: list[StepTelemetryEvent] = field(default_factory=list)

    def __call__(self, event: StepTelemetryEvent, /) -> None:
        """Append the event unchanged."""
        self.events.append(event)

    def as_dicts(self) -> list[dict[str, object]]:
        """Return all events in their serializable form."""
        return [event.as_dict() for event in self.events]
