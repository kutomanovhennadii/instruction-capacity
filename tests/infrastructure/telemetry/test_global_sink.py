"""Mirror tests of the process-wide IEC telemetry sink."""

import logging
from collections.abc import Iterator

import pytest

from iec.infrastructure.logging import PACKAGE_LOGGER_NAME
from iec.infrastructure.telemetry import (
    TelemetryCollector,
    TelemetryContext,
    reset_telemetry_sink,
    set_telemetry_sink,
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


@pytest.fixture(autouse=True)
def isolated_global_sink() -> Iterator[None]:
    """Do not leave a set sink between tests of the process."""
    try:
        yield
    finally:
        reset_telemetry_sink()


def test_global_sink_receives_events() -> None:
    collector = TelemetryCollector()
    set_telemetry_sink(collector.record)

    @telemetry_step(
        "model_generate",
        module_name="mock_provider",
        context_extractor=lambda value: TelemetryContext(model_id="mock", m=value),
    )
    def generate(value: int) -> int:
        return value * 2

    assert generate(4) == 8
    assert len(collector.events) == 1
    assert collector.events[0].operation == "model_generate"
    assert collector.events[0].m == 4


def test_explicit_sink_wins_over_global() -> None:
    """The explicit sink of the decorator gets the event, the global one does not."""
    explicit = TelemetryCollector()
    global_collector = TelemetryCollector()
    set_telemetry_sink(global_collector.record)

    @telemetry_step("run_point", module_name="orchestrator", sink=explicit.record)
    def run() -> None:
        return None

    run()

    assert len(explicit.events) == 1
    assert global_collector.events == []


def test_reset_restores_silence() -> None:
    collector = TelemetryCollector()
    set_telemetry_sink(collector.record)
    reset_telemetry_sink()

    @telemetry_step("run_point", module_name="orchestrator")
    def run() -> None:
        return None

    run()

    assert collector.events == []


def test_no_sinks_keeps_logging_only(capsys: pytest.CaptureFixture[str]) -> None:
    """Without sinks the behaviour is the same as before: the event goes to the log only."""

    @telemetry_step("health_check", module_name="probe")
    def probe() -> str:
        return "ok"

    assert probe() == "ok"
    captured = capsys.readouterr()
    assert "[module=probe step=health_check]" in captured.err
    assert "step status=success" in captured.err
