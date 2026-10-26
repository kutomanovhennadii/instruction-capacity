"""Mirror tests of the raw model answer."""

from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import RawOutput, TokenUsage


def test_technical_fields_are_separate_from_text() -> None:
    raw_provider_response = {"provider_id": "response-1"}
    output = RawOutput(
        text="exact raw text",
        usage=TokenUsage(prompt_tokens=10, response_tokens=3),
        finish_reason="stop",
        latency_s=0.125,
        raw=raw_provider_response,
    )

    assert output.as_dict() == {
        "text": "exact raw text",
        "usage": {"prompt_tokens": 10, "response_tokens": 3},
        "finish_reason": "stop",
        "latency_s": 0.125,
        "raw": raw_provider_response,
    }


def test_optional_technical_fields_default_to_none() -> None:
    output = RawOutput(text="answer")

    assert output.usage is None
    assert output.finish_reason is None
    assert output.latency_s is None
    assert output.raw is None


def test_usage_is_a_project_contract_not_a_provider_dict() -> None:
    """The counter is brought to project fields at the adapter boundary: no provider names."""
    output = RawOutput(text="answer", usage=TokenUsage(prompt_tokens=1))

    assert isinstance(output.usage, TokenUsage)
    assert output.as_dict()["usage"] == {"prompt_tokens": 1, "response_tokens": None}


def test_contract_is_frozen() -> None:
    output = RawOutput(text="answer")

    with pytest.raises(FrozenInstanceError):
        output.text = "changed"  # type: ignore[misc]
