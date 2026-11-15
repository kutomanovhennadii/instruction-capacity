"""Mirror tests of the conditions of one measurement point."""

from iec.contracts import PointConditions

_CONDITIONS = PointConditions(D=40, n=3, m=8, seed_rules=7, seed_text=11, seed_order=4)


def test_text_length_is_distinct_words_times_occurrences() -> None:
    assert _CONDITIONS.v == 120


def test_identity_lists_the_six_numbers_in_order() -> None:
    assert _CONDITIONS.identity == (40, 3, 8, 7, 11, 4)


def test_dict_holds_the_six_numbers() -> None:
    assert _CONDITIONS.as_dict() == {
        "D": 40,
        "n": 3,
        "m": 8,
        "seed_rules": 7,
        "seed_text": 11,
        "seed_order": 4,
    }


def test_seed_order_defaults_to_the_drawn_order() -> None:
    assert PointConditions(D=4, n=5, m=1, seed_rules=1, seed_text=2).seed_order == 0
