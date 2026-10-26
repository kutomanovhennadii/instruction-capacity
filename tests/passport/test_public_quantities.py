"""Tests of the public quantities module of the VS-PMR passport."""

from iec.passport.public_quantities import PUBLIC_KEYS, Quantity


def test_a_quantity_survives_the_json_round_trip() -> None:
    quantity = Quantity("iec_capacity", 56.6, 41.4, "measured", 0.95, "bins", {"support": 512})
    assert Quantity.from_dict(quantity.as_dict()) == quantity


def test_the_public_keys_have_no_duplicates() -> None:
    assert len(PUBLIC_KEYS) == len(set(PUBLIC_KEYS))
