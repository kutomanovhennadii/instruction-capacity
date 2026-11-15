"""Mirror tests of the telemetry event contracts and of the point context."""

import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.infrastructure.telemetry.telemetry_events import (
    TELEMETRY_CONTEXT_INVALID,
    TELEMETRY_FORMAT_VERSION,
    ProviderFields,
    StepTelemetryEvent,
    TelemetryContext,
    TelemetryStatus,
    point_context,
)

EventFactory = Callable[..., StepTelemetryEvent]


def test_log_fields_follow_the_point_order_and_the_log_names() -> None:
    context = TelemetryContext(model_id="mock-1", m=8, len_tokens=4096, seed_rules=0)

    assert context.log_fields() == (
        ("model", "mock-1"),
        ("m", 8),
        ("len", 4096),
        ("seed_rules", 0),
    )


def test_point_context_takes_the_first_present_value() -> None:
    rule_table = SimpleNamespace(m=8, seed_rules=7, seed_text=None)
    conditions = SimpleNamespace(m=99, seed_text=11, seed_order=4)

    assert point_context(rule_table, conditions) == TelemetryContext(
        m=8, seed_rules=7, seed_text=11, seed_order=4
    )


def test_point_context_takes_the_model_identifier() -> None:
    client = SimpleNamespace(model_id="mock-1")

    assert point_context(client, SimpleNamespace(m=4)) == TelemetryContext(model_id="mock-1", m=4)


def test_point_context_of_objects_without_point_fields_is_empty() -> None:
    assert point_context(object(), SimpleNamespace(name="x")) == TelemetryContext()


@pytest.mark.parametrize(
    "source",
    [SimpleNamespace(m="8"), SimpleNamespace(m=True), SimpleNamespace(model_id=5)],
)
def test_point_context_refuses_a_wrong_type(source: SimpleNamespace) -> None:
    with pytest.raises(ContractViolationError) as caught:
        point_context(source)

    assert caught.value.error_code == TELEMETRY_CONTEXT_INVALID


def test_event_dict_is_flat_and_versioned(make_event: EventFactory) -> None:
    event = make_event(
        context=TelemetryContext(model_id="mock-1", m=8),
        provider=ProviderFields(usage={"prompt_tokens": 3}, finish_reason="stop"),
    )

    assert event.as_dict() == {
        "format_version": TELEMETRY_FORMAT_VERSION,
        "block": "model_call",
        "module_name": "mock_provider",
        "operation": "model_generate",
        "status": "success",
        "started_at": 10.0,
        "ended_at": 10.5,
        "duration_s": 0.5,
        "model_id": "mock-1",
        "m": 8,
        "len_tokens": None,
        "seed_rules": None,
        "seed_text": None,
        "seed_order": None,
        "error_code": None,
        "error_type": None,
        "error_message": None,
        "usage": {"prompt_tokens": 3},
        "finish_reason": "stop",
    }


def test_event_json_line_keeps_unicode_and_round_trips(make_event: EventFactory) -> None:
    event = make_event(status=TelemetryStatus.FAILURE, error_message="café closed")

    line = event.as_json_line()

    assert "café" in line
    assert "\n" not in line
    assert json.loads(line) == event.as_dict()


def test_event_is_immutable(make_event: EventFactory) -> None:
    with pytest.raises(AttributeError):
        setattr(make_event(), "status", TelemetryStatus.FAILURE)
