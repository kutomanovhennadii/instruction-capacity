"""Typed context of IEC log records: the module, the step and the identifiers of a point."""

from __future__ import annotations

from dataclasses import dataclass

#: One identifier of a log record: its name in the prefix and its value.
LogField = tuple[str, str | int]


@dataclass(frozen=True, slots=True)
class LogContext:
    """Context of one logical operation, rendered as a stable prefix of its log records.

    ``module_name`` is always present. ``step_name`` names the operation, and ``fields`` are the
    identifiers of the measurement point in the order the prefix shows them. The telemetry layer
    builds them from the point context (``TelemetryContext.log_fields``), so the names of the
    point fields live in one place.
    """

    module_name: str
    step_name: str | None = None
    fields: tuple[LogField, ...] = ()

    def render(self) -> str:
        """Return the context as a ``[key=value ...]`` prefix: module, step, then the fields."""
        parts = [f"module={self.module_name}"]
        if self.step_name:
            parts.append(f"step={self.step_name}")
        parts.extend(f"{name}={value}" for name, value in self.fields)
        return "[" + " ".join(parts) + "]"
