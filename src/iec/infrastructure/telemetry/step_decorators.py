"""Telemetry decorator of the working operations of the IEC bench."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry.global_sink import get_telemetry_sink
from iec.infrastructure.telemetry.telemetry_events import (
    StepTelemetryEvent,
    TelemetryContext,
    TelemetryStatus,
)

P = ParamSpec("P")
R = TypeVar("R")

ContextExtractor = Callable[..., TelemetryContext | None]
TelemetrySink = Callable[[StepTelemetryEvent], None]


def _default_module_name(func: Callable[..., object]) -> str:
    """Derive a stable module or class name of the wrapped function."""
    parts = func.__qualname__.split(".")
    container = [part for part in parts[:-1] if part != "<locals>"]
    if container:
        return container[-1]
    return func.__module__.rsplit(".", 1)[-1]


def _emit(event: StepTelemetryEvent, sink: TelemetrySink | None) -> None:
    """Log the event and pass it to the explicit sink or to the global sink."""
    logger = get_logger(
        LogContext(
            module_name=event.module_name,
            step_name=event.operation,
            model_id=event.model_id,
            m=event.m,
            len_tokens=event.len_tokens,
            seed_rules=event.seed_rules,
            seed_text=event.seed_text,
            seed_order=event.seed_order,
        )
    )
    summary = f"step status={event.status.value} duration_s={event.duration_s:.6f}"
    if event.status is TelemetryStatus.FAILURE:
        logger.error(
            f"{summary} error_code={event.error_code} "
            f"error_type={event.error_type} error_message={event.error_message}"
        )
    else:
        logger.info(summary)
    effective = sink if sink is not None else get_telemetry_sink()
    if effective is not None:
        effective(event)


def telemetry_step(
    operation: str,
    *,
    module_name: str | None = None,
    context_extractor: ContextExtractor | None = None,
    sink: TelemetrySink | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record the telemetry of a successful or failed call of an operation.

    Args:
        operation: Name of the operation in the event.
        module_name: Explicit module name; without it the name is derived from the
            function.
        context_extractor: A function receiving the same arguments and returning the
            context.
        sink: Optional event handler in addition to logging.

    Returns:
        A decorator that keeps the signature and the return value of the function.

    Raises:
        Re-raises any exception of the wrapped operation unchanged after recording an
        event with status ``failure``.

    Side effects:
        Logs one event per call and passes it to ``sink`` when one is given.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        resolved_module = module_name if module_name is not None else _default_module_name(func)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            context = context_extractor(*args, **kwargs) if context_extractor is not None else None
            started_at = time.time()
            started_perf = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                duration_s = time.perf_counter() - started_perf
                _emit(
                    _build_event(
                        resolved_module,
                        operation,
                        TelemetryStatus.FAILURE,
                        started_at,
                        duration_s,
                        context,
                        error=error,
                    ),
                    sink,
                )
                raise
            duration_s = time.perf_counter() - started_perf
            _emit(
                _build_event(
                    resolved_module,
                    operation,
                    TelemetryStatus.SUCCESS,
                    started_at,
                    duration_s,
                    context,
                    error=None,
                ),
                sink,
            )
            return result

        return wrapper

    return decorator


def _build_event(
    module_name: str,
    operation: str,
    status: TelemetryStatus,
    started_at: float,
    duration_s: float,
    context: TelemetryContext | None,
    *,
    error: Exception | None,
) -> StepTelemetryEvent:
    """Assemble the event from timing, context and the optional error."""
    return StepTelemetryEvent(
        module_name=module_name,
        operation=operation,
        status=status,
        started_at=started_at,
        ended_at=started_at + duration_s,
        duration_s=duration_s,
        model_id=context.model_id if context is not None else None,
        m=context.m if context is not None else None,
        len_tokens=context.len_tokens if context is not None else None,
        seed_rules=context.seed_rules if context is not None else None,
        seed_text=context.seed_text if context is not None else None,
        seed_order=context.seed_order if context is not None else None,
        error_code=getattr(error, "error_code", None) if error is not None else None,
        error_type=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
    )
