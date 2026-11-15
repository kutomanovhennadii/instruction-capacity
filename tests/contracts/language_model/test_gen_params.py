"""Mirror tests of the generation parameters."""

from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import GenParams


def test_defaults_and_as_dict() -> None:
    params = GenParams()

    assert params.as_dict() == {
        "temperature": 0.0,
        "max_tokens": 1024,
        "top_p": 1.0,
        "seed": None,
    }


def test_explicit_values_are_preserved() -> None:
    params = GenParams(temperature=0.25, max_tokens=64, top_p=0.9, seed=17)

    assert params.as_dict()["seed"] == 17
    assert params.as_dict()["temperature"] == 0.25


def test_contract_is_frozen() -> None:
    params = GenParams()

    with pytest.raises(FrozenInstanceError):
        params.seed = 2  # type: ignore[misc]
