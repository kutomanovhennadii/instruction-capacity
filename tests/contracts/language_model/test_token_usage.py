"""Mirror tests of the token counter of a model call."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import TokenUsage


def test_fields_are_stored_as_given() -> None:
    usage = TokenUsage(prompt_tokens=120, response_tokens=48)

    assert usage.prompt_tokens == 120
    assert usage.response_tokens == 48


def test_missing_counters_default_to_none() -> None:
    # None means "the provider did not report", not zero tokens.
    usage = TokenUsage()

    assert usage.prompt_tokens is None
    assert usage.response_tokens is None


def test_as_dict_is_json_serializable_round_trip() -> None:
    snapshot = TokenUsage(prompt_tokens=120, response_tokens=48).as_dict()

    assert snapshot == {"prompt_tokens": 120, "response_tokens": 48}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_contract_is_frozen() -> None:
    usage = TokenUsage(prompt_tokens=1, response_tokens=1)

    with pytest.raises(FrozenInstanceError):
        usage.prompt_tokens = 2  # type: ignore[misc]
