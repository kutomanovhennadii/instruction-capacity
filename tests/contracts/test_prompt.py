"""Mirror tests of the assembled prompt."""

from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import Prompt


def test_payload_position_defaults_to_none() -> None:
    prompt = Prompt(text="Rewrite the text.")

    assert prompt.as_dict() == {
        "text": "Rewrite the text.",
        "payload_position": None,
    }


def test_payload_position_is_preserved() -> None:
    prompt = Prompt(text="payload then rules", payload_position="BEFORE")

    assert prompt.as_dict()["payload_position"] == "BEFORE"


def test_contract_is_frozen() -> None:
    prompt = Prompt(text="text")

    with pytest.raises(FrozenInstanceError):
        prompt.text = "changed"  # type: ignore[misc]
