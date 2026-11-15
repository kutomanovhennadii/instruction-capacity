"""Mirror tests of the memory sink."""

from collections.abc import Callable

from iec.infrastructure.telemetry.sinks import MemorySink
from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent

EventFactory = Callable[..., StepTelemetryEvent]


def test_memory_sink_keeps_events_in_order(make_event: EventFactory) -> None:
    sink = MemorySink()

    sink(make_event(operation="first"))
    sink(make_event(operation="second"))

    assert [event.operation for event in sink.events] == ["first", "second"]


def test_memory_sink_returns_serializable_events(make_event: EventFactory) -> None:
    sink = MemorySink()

    sink(make_event(operation="fit_bins"))

    assert sink.as_dicts()[0]["operation"] == "fit_bins"
    assert sink.as_dicts()[0]["status"] == "success"
