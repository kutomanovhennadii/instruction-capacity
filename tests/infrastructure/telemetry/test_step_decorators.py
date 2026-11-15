"""Mirror tests of the telemetry decorator of observed objects."""

import logging
from dataclasses import dataclass, field

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
    telemetry_step,
)


@dataclass
class Probe:
    """An observed object: it keeps its hub and remembers the calls of its context extractor."""

    telemetry: TelemetryHub
    extracted: list[int] = field(default_factory=list)

    @telemetry_step(
        "model_generate",
        block=MODEL_CALL,
        module_name="mock_provider",
        context_extractor=lambda self, value: self.context_of(value),
        provider_extractor=lambda result: ProviderFields(usage={"total_tokens": result}),
    )
    def generate(self, value: int) -> int:
        """Double the value, or fail on a negative one."""
        if value < 0:
            raise ProviderError("provider unavailable")
        return value * 2

    @telemetry_step("probe", block=MODEL_CALL)
    def plain(self) -> str:
        """An operation without a context and with the default module name."""
        return "ok"

    def context_of(self, value: int) -> TelemetryContext:
        """The point context of a call; the call is remembered."""
        self.extracted.append(value)
        return TelemetryContext(model_id="mock", m=value)


def _probe(level: ObservationLevel = ObservationLevel.INFO) -> tuple[Probe, MemorySink]:
    sink = MemorySink()
    hub = TelemetryHub(ObservationLevels(blocks={MODEL_CALL: level}), [sink])
    return Probe(hub), sink


def test_success_records_event_and_preserves_result(
    isolated_package_logger: logging.Logger,
) -> None:
    probe, sink = _probe()

    assert probe.generate(4) == 8

    event = sink.events[0]
    assert event.status is TelemetryStatus.SUCCESS
    assert event.block == MODEL_CALL
    assert event.module_name == "mock_provider"
    assert event.operation == "model_generate"
    assert event.context == TelemetryContext(model_id="mock", m=4)
    assert event.provider.usage == {"total_tokens": 8}


def test_failure_records_error_and_reraises_the_exception(
    isolated_package_logger: logging.Logger,
) -> None:
    probe, sink = _probe()

    with pytest.raises(ProviderError) as caught:
        probe.generate(-1)

    assert caught.value.message == "provider unavailable"
    event = sink.events[0]
    assert event.status is TelemetryStatus.FAILURE
    assert event.error_code == "PROVIDER_ERROR"


def test_block_off_skips_timing_and_the_context_extractor(
    isolated_package_logger: logging.Logger,
) -> None:
    probe, sink = _probe(ObservationLevel.OFF)

    assert probe.generate(4) == 8
    assert sink.events == []
    assert probe.extracted == []


def test_module_name_defaults_to_the_class_name(isolated_package_logger: logging.Logger) -> None:
    probe, sink = _probe()

    assert probe.plain() == "ok"
    assert sink.events[0].module_name == "Probe"
    assert sink.events[0].context == TelemetryContext()


def test_decorator_keeps_the_name_and_description_of_the_method() -> None:
    assert Probe.generate.__name__ == "generate"
    assert Probe.generate.__doc__ == "Double the value, or fail on a negative one."
