"""Contracts of the telemetry event and of the in-memory collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class TelemetryStatus(StrEnum):
    """Outcome of a call, decided by the presence or absence of an exception."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(slots=True)
class TelemetryContext:
    """Optional identifiers of a measurement point for a telemetry event."""

    model_id: str | None = None
    m: int | None = None
    len_tokens: int | None = None
    seed_rules: int | None = None
    seed_text: int | None = None
    seed_order: int | None = None


@dataclass(slots=True)
class StepTelemetryEvent:
    """One recorded working operation of the bench.

    Subject statuses of a point are not part of this contract: the decorator tells apart
    only a normal return and an exception. Provider fields are optional and will be filled
    by the model adapters when they appear.
    """

    module_name: str
    operation: str
    status: TelemetryStatus
    started_at: float
    ended_at: float
    duration_s: float
    model_id: str | None = None
    m: int | None = None
    len_tokens: int | None = None
    seed_rules: int | None = None
    seed_text: int | None = None
    seed_order: int | None = None
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    usage: dict[str, int] | None = None
    finish_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the serializable form with the status as a string."""
        return {
            "module_name": self.module_name,
            "operation": self.operation,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "model_id": self.model_id,
            "m": self.m,
            "len_tokens": self.len_tokens,
            "seed_rules": self.seed_rules,
            "seed_text": self.seed_text,
            "seed_order": self.seed_order,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


@dataclass(slots=True)
class TelemetryCollector:
    """Optional sink that collects the events of one run in memory."""

    events: list[StepTelemetryEvent] = field(default_factory=list)

    def record(self, event: StepTelemetryEvent) -> None:
        """Append the event to the collector unchanged."""
        self.events.append(event)

    def as_dicts(self) -> list[dict[str, object]]:
        """Return all events in serializable form."""
        return [event.as_dict() for event in self.events]
