"""Contracts of a telemetry event: the point context, the provider fields and the event.

An event records one call of a working operation: the block of observation it belongs to, the
module and the operation, the outcome, the times and the identifiers of the measurement point.
The serialized form carries ``format_version``, so a reader of telemetry files tells the
versions apart (specification §5: every data format has a version).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TypeVar

from iec.infrastructure.exceptions import ContractViolationError

#: Version of the serialized form of an event; it grows when a field changes its meaning.
TELEMETRY_FORMAT_VERSION = 1
#: Error code of a point identifier of a wrong type.
TELEMETRY_CONTEXT_INVALID = "TELEMETRY_CONTEXT_INVALID"
#: Names of the point fields in the log prefix, where they differ from the field names.
_LOG_NAMES = {"model_id": "model", "len_tokens": "len"}

T = TypeVar("T", str, int)


class TelemetryStatus(StrEnum):
    """Outcome of a call, decided by the presence or absence of an exception."""

    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class TelemetryContext:
    """Identifiers of a measurement point for a telemetry event and its log line.

    Every field is optional: an operation names only what it knows. The order of the fields is
    the order of the point key in the log prefix.
    """

    model_id: str | None = None
    m: int | None = None
    len_tokens: int | None = None
    seed_rules: int | None = None
    seed_text: int | None = None
    seed_order: int | None = None

    def log_fields(self) -> tuple[tuple[str, str | int], ...]:
        """Return the set identifiers as ``(name, value)`` pairs of the log prefix."""
        return tuple(
            (_LOG_NAMES.get(name, name), value)
            for name, value in asdict(self).items()
            if value is not None
        )


def point_context(*sources: object) -> TelemetryContext:
    """Collect the point context from the attributes of the given objects.

    Every field of ``TelemetryContext`` is looked up by its name in the sources in the given
    order, and the first value that is not ``None`` is taken: ``point_context(rule_table,
    conditions)`` takes ``m`` from the rule table and ``seed_text`` from the conditions. One
    function serves every observed operation, so the names of the point fields are written once.

    Args:
        sources: Contracts or other objects with point identifiers among their attributes.

    Returns:
        The context; an identifier that no source has stays ``None``.

    Raises:
        ContractViolationError: an identifier has a wrong type - ``model_id`` must be a string,
            the others integers (``error_code="TELEMETRY_CONTEXT_INVALID"``).
    """
    return TelemetryContext(
        model_id=_first_value(sources, "model_id", str),
        m=_first_value(sources, "m", int),
        len_tokens=_first_value(sources, "len_tokens", int),
        seed_rules=_first_value(sources, "seed_rules", int),
        seed_text=_first_value(sources, "seed_text", int),
        seed_order=_first_value(sources, "seed_order", int),
    )


def _first_value(sources: tuple[object, ...], name: str, kind: type[T]) -> T | None:
    """Return the first value of the attribute ``name`` that is not ``None``, of type ``kind``.

    Raises:
        ContractViolationError: the value is not of type ``kind``; a boolean is not an integer.
    """
    for source in sources:
        value = getattr(source, name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ContractViolationError(
                f"point field '{name}' of {type(source).__name__} is {value!r}, "
                f"not {kind.__name__}",
                error_code=TELEMETRY_CONTEXT_INVALID,
            )
        return value
    return None


@dataclass(frozen=True, slots=True)
class ProviderFields:
    """Technical fields of a model provider in the event of a model call.

    ``usage`` holds the token counts of the call and ``finish_reason`` the reason the provider
    stopped; both stay ``None`` for operations that are not model calls.
    """

    usage: dict[str, int] | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class StepTelemetryEvent:
    """One recorded call of a working operation of the bench.

    The subject status of a point (``ok``, ``na``, ``api_error``) is not part of this contract:
    the event tells apart a normal return and an exception only.
    """

    block: str
    module_name: str
    operation: str
    status: TelemetryStatus
    started_at: float
    ended_at: float
    duration_s: float
    context: TelemetryContext = field(default_factory=TelemetryContext)
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    provider: ProviderFields = field(default_factory=ProviderFields)

    def as_dict(self) -> dict[str, object]:
        """Return the flat serializable form: version, block, call, point, error, provider."""
        return {
            "format_version": TELEMETRY_FORMAT_VERSION,
            "block": self.block,
            "module_name": self.module_name,
            "operation": self.operation,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            **asdict(self.context),
            "error_code": self.error_code,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **asdict(self.provider),
        }

    def as_json_line(self) -> str:
        """Return the serializable form as one JSON line without a line end, text in Unicode."""
        return json.dumps(self.as_dict(), ensure_ascii=False)
