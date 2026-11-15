"""Telemetry decorator of the working operations of the IEC bench.

``@telemetry_step`` wraps a method of an observed object: a class whose instances keep the
telemetry hub in the attribute ``telemetry``, handed to them through the constructor. At every
call the decorator takes the hub of the instance and records the call through
``TelemetryHub.observed``, so there is no process-wide sink. The decorator never swallows an
exception: it records the failure and re-raises the same exception (specification §8).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

from iec.infrastructure.telemetry.telemetry_events import ProviderFields, TelemetryContext
from iec.infrastructure.telemetry.telemetry_hub import TelemetryHub

P = ParamSpec("P")
R = TypeVar("R")

#: Function of the same arguments as the method, the instance first, returning the point context.
ContextExtractor = Callable[..., TelemetryContext | None]
#: Function of the result of the method returning the provider fields of a model call.
ProviderExtractor = Callable[[Any], ProviderFields | None]


class Observed(Protocol):
    """An object whose methods are observed: it keeps its telemetry hub."""

    @property
    def telemetry(self) -> TelemetryHub:
        """The telemetry hub handed to the object through its constructor."""
        ...


ObservedT = TypeVar("ObservedT", bound=Observed)


def telemetry_step(
    operation: str,
    *,
    block: str,
    module_name: str | None = None,
    context_extractor: ContextExtractor | None = None,
    provider_extractor: ProviderExtractor | None = None,
) -> Callable[[Callable[Concatenate[ObservedT, P], R]], Callable[Concatenate[ObservedT, P], R]]:
    """Record the telemetry of every call of a method of an observed object.

    Args:
        operation: Name of the operation in the event.
        block: Block of observation the operation belongs to; its level in the hub decides
            whether the call is recorded.
        module_name: Explicit module name; without it the name of the class is taken.
        context_extractor: Function receiving the instance and the arguments of the call and
            returning the point context; it is not called when the block is ``off``.
        provider_extractor: Function receiving the result and returning the provider fields of a
            model call.

    Returns:
        A decorator that keeps the signature and the return value of the method.

    Raises:
        Re-raises any exception of the method unchanged after recording a failure event.

    Side effects:
        Records one event per call in the hub of the instance.
    """

    def decorator(
        method: Callable[Concatenate[ObservedT, P], R],
    ) -> Callable[Concatenate[ObservedT, P], R]:
        """Wrap one method of an observed class."""
        resolved_module = module_name if module_name is not None else _default_module_name(method)

        @functools.wraps(method)
        def wrapper(observed: ObservedT, /, *args: P.args, **kwargs: P.kwargs) -> R:
            """Call the method and record the call when its block is observed."""
            hub = observed.telemetry
            if not hub.observes(block):
                return method(observed, *args, **kwargs)
            context: TelemetryContext | None = None
            if context_extractor is not None:
                context = context_extractor(observed, *args, **kwargs)
            with hub.observed(
                block, operation, module_name=resolved_module, context=context
            ) as call:
                result = method(observed, *args, **kwargs)
                if provider_extractor is not None:
                    fields = provider_extractor(result)
                    if fields is not None:
                        call.provider = fields
                return result

        return wrapper

    return decorator


def _default_module_name(method: Callable[..., object]) -> str:
    """Derive a stable class or module name of the wrapped method."""
    parts = method.__qualname__.split(".")
    container = [part for part in parts[:-1] if part != "<locals>"]
    if container:
        return container[-1]
    return method.__module__.rsplit(".", 1)[-1]
