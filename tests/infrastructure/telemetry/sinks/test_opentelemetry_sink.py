"""Mirror tests of the OpenTelemetry sink with a tracer double: OpenTelemetry is not installed."""

from collections.abc import Callable, Mapping

from iec.infrastructure.telemetry.sinks import OpenTelemetrySink, span_attributes
from iec.infrastructure.telemetry.telemetry_events import (
    ProviderFields,
    StepTelemetryEvent,
    TelemetryContext,
    TelemetryStatus,
)

EventFactory = Callable[..., StepTelemetryEvent]


class FakeSpan:
    """A span double that remembers its name, attributes, times and statuses."""

    def __init__(self, name: str, attributes: Mapping[str, object] | None, start: int | None):
        self.name = name
        self.attributes = dict(attributes or {})
        self.start_time = start
        self.end_time: int | None = None
        self.statuses: list[object] = []

    def set_status(self, status: object, /) -> None:
        self.statuses.append(status)

    def end(self, end_time: int | None = None) -> None:
        self.end_time = end_time


class FakeTracer:
    """A tracer double that keeps the spans it started."""

    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, object] | None = None,
        start_time: int | None = None,
    ) -> FakeSpan:
        span = FakeSpan(name, attributes, start_time)
        self.spans.append(span)
        return span


def _error_status(message: str) -> tuple[str, str]:
    return ("error", message)


def test_successful_call_becomes_a_finished_span(make_event: EventFactory) -> None:
    tracer = FakeTracer()
    sink = OpenTelemetrySink(tracer, _error_status)

    sink(make_event(context=TelemetryContext(model_id="mock-1", m=8)))

    span = tracer.spans[0]
    assert span.name == "model_call.model_generate"
    assert span.start_time == 10_000_000_000
    assert span.end_time == 10_500_000_000
    assert span.attributes["iec.model_id"] == "mock-1"
    assert span.attributes["iec.m"] == 8
    assert span.attributes["iec.status"] == "success"
    assert span.statuses == []


def test_failed_call_gets_the_error_status(make_event: EventFactory) -> None:
    tracer = FakeTracer()
    sink = OpenTelemetrySink(tracer, _error_status)

    sink(
        make_event(
            status=TelemetryStatus.FAILURE,
            error_type="ProviderError",
            error_message="provider unavailable",
        )
    )

    assert tracer.spans[0].statuses == [("error", "provider unavailable")]


def test_attributes_leave_out_unset_fields_and_flatten_usage(make_event: EventFactory) -> None:
    usage = ProviderFields(usage={"prompt_tokens": 3, "completion_tokens": 5})

    attributes = span_attributes(make_event(provider=usage))

    assert attributes["iec.usage.prompt_tokens"] == 3
    assert attributes["iec.usage.completion_tokens"] == 5
    assert attributes["iec.format_version"] == 1
    assert "iec.error_code" not in attributes
    assert "iec.finish_reason" not in attributes
