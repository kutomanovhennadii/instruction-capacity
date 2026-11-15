"""Mirror tests of the telemetry hub."""

import logging

import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.infrastructure.telemetry import (
    MODEL_CALL,
    MemorySink,
    ObservationLevel,
    ObservationLevels,
    ProviderFields,
    TelemetryContext,
    TelemetryHub,
    TelemetryStatus,
)


def _hub(sink: MemorySink, **levels: ObservationLevel) -> TelemetryHub:
    return TelemetryHub(ObservationLevels(blocks=levels), [sink])


def test_observed_call_records_a_success_with_its_context(
    isolated_package_logger: logging.Logger,
) -> None:
    sink = MemorySink()
    context = TelemetryContext(m=8)

    with _hub(sink).observed(MODEL_CALL, "model_generate", module_name="mock", context=context):
        pass

    event = sink.events[0]
    assert event.status is TelemetryStatus.SUCCESS
    assert event.block == MODEL_CALL
    assert event.module_name == "mock"
    assert event.context.m == 8
    assert event.duration_s >= 0
    assert event.ended_at >= event.started_at


def test_observed_call_records_a_failure_and_reraises_the_same_exception(
    isolated_package_logger: logging.Logger,
) -> None:
    sink = MemorySink()
    expected = ProviderError("provider unavailable")

    with pytest.raises(ProviderError) as caught:
        with _hub(sink).observed(MODEL_CALL, "model_generate", module_name="mock"):
            raise expected

    assert caught.value is expected
    event = sink.events[0]
    assert event.status is TelemetryStatus.FAILURE
    assert event.error_code == "PROVIDER_ERROR"
    assert event.error_type == "ProviderError"
    assert event.error_message == "provider unavailable"


def test_block_off_records_nothing(isolated_package_logger: logging.Logger) -> None:
    sink = MemorySink()
    hub = _hub(sink, model_call=ObservationLevel.OFF)

    with hub.observed(MODEL_CALL, "model_generate", module_name="mock"):
        pass

    assert hub.observes(MODEL_CALL) is False
    assert sink.events == []


def test_error_level_keeps_failures_only(isolated_package_logger: logging.Logger) -> None:
    sink = MemorySink()
    hub = _hub(sink, model_call=ObservationLevel.ERROR)

    with hub.observed(MODEL_CALL, "first", module_name="mock"):
        pass
    with pytest.raises(ValueError):
        with hub.observed(MODEL_CALL, "second", module_name="mock"):
            raise ValueError("bad value")

    assert [event.operation for event in sink.events] == ["second"]


def test_provider_fields_of_the_call_reach_the_event(
    isolated_package_logger: logging.Logger,
) -> None:
    sink = MemorySink()

    with _hub(sink).observed(MODEL_CALL, "model_generate", module_name="mock") as call:
        call.provider = ProviderFields(usage={"prompt_tokens": 3}, finish_reason="stop")

    assert sink.events[0].provider.finish_reason == "stop"
    assert sink.events[0].provider.usage == {"prompt_tokens": 3}


def test_attached_sink_receives_events_inside_its_block_only(
    isolated_package_logger: logging.Logger,
) -> None:
    hub = TelemetryHub.log_only()
    attached = MemorySink()

    with hub.attached(attached):
        with hub.observed(MODEL_CALL, "inside", module_name="mock"):
            pass
    with hub.observed(MODEL_CALL, "outside", module_name="mock"):
        pass

    assert [event.operation for event in attached.events] == ["inside"]
    assert hub.sinks == ()


def test_nested_attachments_of_one_sink_unwind_one_by_one() -> None:
    hub = TelemetryHub.log_only()
    sink = MemorySink()

    with hub.attached(sink):
        with hub.attached(sink):
            assert hub.sinks == (sink, sink)
        assert hub.sinks == (sink,)
    assert hub.sinks == ()


def test_log_line_keeps_the_format_of_the_bench(
    isolated_package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = TelemetryHub.log_only()
    context = TelemetryContext(model_id="mock", seed_order=4)

    with hub.observed(MODEL_CALL, "health_check", module_name="probe", context=context):
        pass

    err = capsys.readouterr().err
    assert "[module=probe step=health_check model=mock seed_order=4]" in err
    assert "step status=success duration_s=" in err


def test_failure_log_line_carries_the_error(
    isolated_package_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = TelemetryHub.log_only()

    with pytest.raises(ProviderError):
        with hub.observed(MODEL_CALL, "model_generate", module_name="probe"):
            raise ProviderError("provider unavailable")

    err = capsys.readouterr().err
    assert " ERROR " in err
    assert "step status=failure" in err
    assert "error_code=PROVIDER_ERROR error_type=ProviderError" in err
    assert "error_message=provider unavailable" in err
