"""Mirror tests of the model tariff contract."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import Tariff

_MINI = Tariff(input_usd_per_million=0.15, output_usd_per_million=0.60)


def test_fields_are_stored_as_given() -> None:
    assert _MINI.input_usd_per_million == 0.15
    assert _MINI.output_usd_per_million == 0.60


def test_cost_counts_both_directions_at_their_own_prices() -> None:
    """Input and output are priced differently - the formula must tell them apart."""
    cost = _MINI.cost_usd(prompt_tokens=1_000_000, response_tokens=0)
    assert cost == pytest.approx(0.15)

    cost = _MINI.cost_usd(prompt_tokens=0, response_tokens=1_000_000)
    assert cost == pytest.approx(0.60)

    cost = _MINI.cost_usd(prompt_tokens=2_000_000, response_tokens=500_000)
    assert cost == pytest.approx(2 * 0.15 + 0.5 * 0.60)


def test_cost_of_nothing_is_zero() -> None:
    assert _MINI.cost_usd(prompt_tokens=0, response_tokens=0) == 0.0


def test_realistic_call_costs_a_fraction_of_a_cent() -> None:
    """Order of magnitude: one call of a pilot costs a fraction of a cent, not dollars."""
    cost = _MINI.cost_usd(prompt_tokens=1200, response_tokens=300)

    assert cost == pytest.approx((1200 * 0.15 + 300 * 0.60) / 1e6)
    assert cost < 0.001


def test_as_dict_is_json_serializable_round_trip() -> None:
    snapshot = _MINI.as_dict()

    assert snapshot == {"input_usd_per_million": 0.15, "output_usd_per_million": 0.60}
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_contract_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _MINI.input_usd_per_million = 0.30  # type: ignore[misc]
