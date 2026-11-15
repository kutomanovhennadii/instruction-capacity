"""Mirror tests of the replacement table contract."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import Rule, RuleTable


def _make_table() -> RuleTable:
    """Assemble a table of two rules in a meaningful order."""
    return RuleTable(
        rules=(Rule(source="lake", target="stone"), Rule(source="dog", target="cloud")),
        vocab_id="v1_intersection",
        seed_rules=7,
    )


def test_fields_are_stored_as_given() -> None:
    table = _make_table()

    assert table.rules == (Rule("lake", "stone"), Rule("dog", "cloud"))
    assert table.vocab_id == "v1_intersection"
    assert table.seed_rules == 7


def test_m_equals_rule_count() -> None:
    table = _make_table()

    assert table.m == 2
    assert table.m == len(table.rules)


def test_sources_and_targets_preserve_table_order() -> None:
    table = _make_table()

    assert table.sources == ("lake", "dog")
    assert table.targets == ("stone", "cloud")


def test_as_dict_is_json_serializable_with_rule_order() -> None:
    table = _make_table()

    snapshot = table.as_dict()

    assert snapshot == {
        "rules": [
            {"source": "lake", "target": "stone"},
            {"source": "dog", "target": "cloud"},
        ],
        "vocab_id": "v1_intersection",
        "seed_rules": 7,
        "m": 2,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_empty_table_has_zero_m() -> None:
    table = RuleTable(rules=(), vocab_id="v1_intersection", seed_rules=0)

    assert table.m == 0
    assert table.sources == ()
    assert table.targets == ()


def test_rule_is_frozen() -> None:
    rule = Rule(source="lake", target="stone")

    with pytest.raises(FrozenInstanceError):
        rule.source = "sea"  # type: ignore[misc]


def test_table_is_frozen() -> None:
    table = _make_table()

    with pytest.raises(FrozenInstanceError):
        table.vocab_id = "other"  # type: ignore[misc]
