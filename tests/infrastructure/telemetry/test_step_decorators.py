"""Mirror tests of the IEC telemetry events and decorator."""

import logging
from collections.abc import Iterator

import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.infrastructure.logging import PACKAGE_LOGGER_NAME
from iec.infrastructure.telemetry import (
    StepTelemetryEvent,
    TelemetryCollector,
    TelemetryContext,
    TelemetryStatus,
    telemetry_step,
)


@pytest.fixture(autouse=True)
def isolated_package_logger() -> Iterator[None]:
    """Do not leave global log handlers between tests."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers.clear()
    try:
        yield
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def test_success_records_event_and_preserves_result() -> None:
    collector = TelemetryCollector()

    @telemetry_step(
        "model_generate",
        module_name="mock_provider",
        context_extractor=lambda value: TelemetryContext(model_id="mock", m=value),
        sink=collector.record,
    )
    def generate(value: int) -> int:
        return value * 2

    assert generate(4) == 8
    assert len(collector.events) == 1
    event = collector.events[0]
    assert event.status is TelemetryStatus.SUCCESS
    assert event.module_name == "mock_provider"
    assert event.operation == "model_generate"
    assert event.model_id == "mock"
    assert event.m == 4
    assert event.duration_s >= 0
    assert event.ended_at >= event.started_at
    assert event.error_type is None


def test_permutation_axis_reaches_event_and_log_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The order seed goes context -> event -> log prefix: the point identity is complete."""
    collector = TelemetryCollector()

    @telemetry_step(
        "rules_generate",
        module_name="rule_generator",
        context_extractor=lambda: TelemetryContext(m=8, seed_rules=7, seed_order=4),
        sink=collector.record,
    )
    def generate() -> None:
        return None

    generate()

    assert collector.events[0].seed_order == 4
    assert collector.as_dicts()[0]["seed_order"] == 4
    assert "seed_order=4" in capsys.readouterr().err


def test_failure_records_error_and_reraises_same_exception() -> None:
    collector = TelemetryCollector()
    expected = ProviderError("provider unavailable")

    @telemetry_step("model_generate", sink=collector.record)
    def fail() -> None:
        raise expected

    with pytest.raises(ProviderError) as caught:
        fail()

    assert caught.value is expected
    event = collector.events[0]
    assert event.status is TelemetryStatus.FAILURE
    assert event.error_code == "PROVIDER_ERROR"
    assert event.error_type == "ProviderError"
    assert event.error_message == "provider unavailable"


def test_collector_returns_serializable_events() -> None:
    collector = TelemetryCollector()
    collector.record(
        StepTelemetryEvent(
            module_name="runner",
            operation="run",
            status=TelemetryStatus.SUCCESS,
            started_at=1.0,
            ended_at=1.5,
            duration_s=0.5,
        )
    )

    assert collector.as_dicts()[0]["status"] == "success"
    assert collector.as_dicts()[0]["duration_s"] == 0.5


def test_sink_none_logs_event_without_changing_result(capsys: pytest.CaptureFixture[str]) -> None:
    @telemetry_step("health_check", module_name="probe")
    def probe() -> str:
        return "ok"

    assert probe() == "ok"
    captured = capsys.readouterr()
    assert "[module=probe step=health_check]" in captured.err
    assert "step status=success" in captured.err
