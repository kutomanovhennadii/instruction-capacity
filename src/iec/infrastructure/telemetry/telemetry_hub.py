"""Telemetry hub: routes the events of observed calls to the log and the sinks by block levels.

The hub is built once at the setup of a command - by the observability builder from the
configuration, or as ``TelemetryHub.log_only()`` - and handed to the observed objects through
their constructors, so there is no process-wide state (specification §7.7). For every event it
checks the level of the block, writes the log line and passes the event to every sink.
``observed`` records one call made inside a ``with`` block; ``attached`` adds a sink for the
time of a ``with`` block, so a run can collect its own events without rebuilding the objects it
observes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry.observation_blocks import (
    EVENT_SEVERITY,
    ObservationLevel,
    ObservationLevels,
)
from iec.infrastructure.telemetry.sinks.sink_contract import TelemetrySink
from iec.infrastructure.telemetry.telemetry_events import (
    ProviderFields,
    StepTelemetryEvent,
    TelemetryContext,
    TelemetryStatus,
)


@dataclass(slots=True)
class ObservedCall:
    """Handle of one observed call: the caller may put the provider fields of its result here."""

    provider: ProviderFields = field(default_factory=ProviderFields)


class TelemetryHub:
    """Router of telemetry events: the levels of the blocks, the log line and the sinks."""

    def __init__(self, levels: ObservationLevels, sinks: Sequence[TelemetrySink] = ()) -> None:
        """Remember the levels and the sinks.

        Args:
            levels: Level of every block of observation.
            sinks: Receivers of the events that pass the levels, besides the log.
        """
        self._levels = levels
        self._sinks: tuple[TelemetrySink, ...] = tuple(sinks)
        self._lock = threading.Lock()

    @classmethod
    def log_only(cls) -> TelemetryHub:
        """Return a hub with the default level for every block and no sinks: the log only."""
        return cls(ObservationLevels())

    @property
    def sinks(self) -> tuple[TelemetrySink, ...]:
        """The sinks that receive events now."""
        return self._sinks

    def observes(self, block: str) -> bool:
        """Tell whether the block is observed at all, that is, its level is not ``off``."""
        return self._levels.level_of(block) is not ObservationLevel.OFF

    @contextmanager
    def observed(
        self,
        block: str,
        operation: str,
        *,
        module_name: str,
        context: TelemetryContext | None = None,
    ) -> Iterator[ObservedCall]:
        """Record one call of an operation made inside the ``with`` block.

        Args:
            block: Block of observation of the operation.
            operation: Name of the operation in the event.
            module_name: Name of the module or class that makes the call.
            context: Identifiers of the measurement point, when known.

        Yields:
            The handle of the call, where the caller may put the provider fields.

        Raises:
            Re-raises any exception of the ``with`` block unchanged after recording a failure.

        Side effects:
            Records one event; when the level of the block is ``off``, nothing is timed or
            recorded.
        """
        call = ObservedCall()
        if not self.observes(block):
            yield call
            return
        point = context if context is not None else TelemetryContext()
        started_at = time.time()
        started = time.perf_counter()
        try:
            yield call
        except Exception as error:
            self.record(
                _event(block, module_name, operation, started_at, started, point, call, error)
            )
            raise
        self.record(_event(block, module_name, operation, started_at, started, point, call, None))

    def record(self, event: StepTelemetryEvent) -> None:
        """Pass an event to the log and to the sinks when the level of its block admits it.

        Side effects:
            Writes one log record with the severity of the event and calls every sink.
        """
        if not self._levels.level_of(event.block).admits(event.status):
            return
        _log_event(event)
        for sink in self._sinks:
            sink(event)

    @contextmanager
    def attached(self, sink: TelemetrySink) -> Iterator[None]:
        """Add a sink for the time of the ``with`` block.

        Side effects:
            The sink receives the events recorded inside the block, from any thread; at the exit
            it is removed, also when the block raises.
        """
        with self._lock:
            self._sinks = (*self._sinks, sink)
        try:
            yield
        finally:
            with self._lock:
                self._sinks = _without(self._sinks, sink)


def _event(
    block: str,
    module_name: str,
    operation: str,
    started_at: float,
    started: float,
    context: TelemetryContext,
    call: ObservedCall,
    error: Exception | None,
) -> StepTelemetryEvent:
    """Assemble the event of a finished call.

    Args:
        started_at: Wall-clock start of the call, seconds since the epoch.
        started: Value of the performance counter at the start; the duration counts from it.
        error: The exception of a failed call, ``None`` for a successful one.
    """
    duration_s = time.perf_counter() - started
    return StepTelemetryEvent(
        block=block,
        module_name=module_name,
        operation=operation,
        status=TelemetryStatus.SUCCESS if error is None else TelemetryStatus.FAILURE,
        started_at=started_at,
        ended_at=started_at + duration_s,
        duration_s=duration_s,
        context=context,
        error_code=getattr(error, "error_code", None) if error is not None else None,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
        provider=call.provider,
    )


def _log_event(event: StepTelemetryEvent) -> None:
    """Write the log line of an event with the severity of its outcome."""
    context = LogContext(
        module_name=event.module_name,
        step_name=event.operation,
        fields=event.context.log_fields(),
    )
    message = f"step status={event.status.value} duration_s={event.duration_s:.6f}"
    if event.status is TelemetryStatus.FAILURE:
        message += (
            f" error_code={event.error_code} error_type={event.error_type}"
            f" error_message={event.error_message}"
        )
    get_logger(context).log(EVENT_SEVERITY[event.status], message)


def _without(sinks: tuple[TelemetrySink, ...], sink: TelemetrySink) -> tuple[TelemetrySink, ...]:
    """Return the sinks without the last occurrence of ``sink``: nested attachments unwind."""
    for index in range(len(sinks) - 1, -1, -1):
        if sinks[index] is sink:
            return sinks[:index] + sinks[index + 1:]
    return sinks
