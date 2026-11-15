"""Shared fixtures of the infrastructure tests: a factory of telemetry events."""

from collections.abc import Callable

import pytest

from iec.infrastructure.telemetry.telemetry_events import StepTelemetryEvent, TelemetryStatus


@pytest.fixture
def make_event() -> Callable[..., StepTelemetryEvent]:
    """Return a factory of events of a successful model call; keyword arguments change fields."""

    def factory(**changes: object) -> StepTelemetryEvent:
        fields: dict[str, object] = {
            "block": "model_call",
            "module_name": "mock_provider",
            "operation": "model_generate",
            "status": TelemetryStatus.SUCCESS,
            "started_at": 10.0,
            "ended_at": 10.5,
            "duration_s": 0.5,
        }
        fields.update(changes)
        return StepTelemetryEvent(**fields)

    return factory
