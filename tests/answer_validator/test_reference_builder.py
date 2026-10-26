"""Mirror tests of the reference builder."""

import dataclasses
from typing import Any

import pytest

from iec.answer_validator import build_reference
from iec.contracts import Rule, RuleTable, SourceText
from iec.infrastructure.exceptions import ContractViolationError


def _handmade() -> tuple[RuleTable, SourceText]:
    """Build a handmade task: two lines of three words, two targets in the markup."""
    words = ("tree", "sun", "dog", "lake", "tree", "sun")
    table = RuleTable(
        rules=(Rule(source="tree", target="cat"), Rule(source="sun", target="rock")),
        vocab_id="v_test",
        seed_rules=7,
    )
    text = SourceText(
        words=words,
        words_per_line=3,
        replacement_positions=(0, 1, 4, 5),
        len_tokens=len(words),
        tokenizer_name="mock",
        seed_text=42,
    )
    return table, text


def test_targets_replace_sources_and_background_is_untouched() -> None:
    table, text = _handmade()

    reference = build_reference(table, text)

    assert reference.words == ("cat", "rock", "dog", "lake", "cat", "rock")


def test_reference_is_deterministic() -> None:
    table, text = _handmade()

    assert build_reference(table, text) == build_reference(table, text)


def test_reference_keeps_line_width_of_the_source() -> None:
    table, text = _handmade()

    reference = build_reference(table, text)

    assert reference.words_per_line == text.words_per_line
    assert reference.text == "cat rock dog\nlake cat rock"


def test_changed_positions_are_exactly_the_marked_ones(task: Any) -> None:
    reference = build_reference(task.rule_table, task.source_text)

    changed = tuple(
        index
        for index, (before, after) in enumerate(zip(task.source_text.words, reference.words))
        if before != after
    )
    assert changed == task.source_text.replacement_positions
    assert len(reference.words) == task.params.v


def test_every_marked_position_carries_its_rule_target(task: Any) -> None:
    reference = build_reference(task.rule_table, task.source_text)
    mapping = {rule.source: rule.target for rule in task.rule_table.rules}

    for index in task.source_text.replacement_positions:
        assert reference.words[index] == mapping[task.source_text.words[index]]


def test_replacement_words_never_appear_outside_marked_positions(task: Any) -> None:
    """Replacement words are absent from the stream, so they cannot be outside the markup."""
    reference = build_reference(task.rule_table, task.source_text)
    marked = set(task.source_text.replacement_positions)
    targets = frozenset(task.rule_table.targets)

    assert not any(
        word in targets for index, word in enumerate(reference.words) if index not in marked
    )


@pytest.mark.parametrize(
    ("distinct", "appearances", "m", "seed_text"),
    [(6, 10, 2, 42), (6, 10, 3, 7), (12, 10, 5, 1), (8, 5, 4, 99)],
)
def test_builder_holds_across_the_parameter_grid(
    make_task: Any, distinct: int, appearances: int, m: int, seed_text: int
) -> None:
    task = make_task(D=distinct, n=appearances, m=m, seed_text=seed_text)

    reference = build_reference(task.rule_table, task.source_text)

    assert len(reference.words) == distinct * appearances
    assert len(task.source_text.replacement_positions) == m * appearances


def test_markup_pointing_outside_the_stream_is_rejected() -> None:
    table, text = _handmade()
    broken = dataclasses.replace(text, replacement_positions=(0, 1, 4, 99))

    with pytest.raises(ContractViolationError) as error:
        build_reference(table, broken)

    assert error.value.error_code == "REFERENCE_MARKUP_MISMATCH"


def test_markup_pointing_at_a_background_word_is_rejected() -> None:
    table, text = _handmade()
    broken = dataclasses.replace(text, replacement_positions=(0, 1, 3, 4, 5))

    with pytest.raises(ContractViolationError) as error:
        build_reference(table, broken)

    assert error.value.error_code == "REFERENCE_MARKUP_MISMATCH"


def test_missing_markup_of_a_replaced_word_is_rejected() -> None:
    table, text = _handmade()
    broken = dataclasses.replace(text, replacement_positions=(0, 1, 4))

    with pytest.raises(ContractViolationError) as error:
        build_reference(table, broken)

    assert error.value.error_code == "REFERENCE_MARKUP_MISMATCH"


def test_identity_rule_is_rejected_as_unobservable() -> None:
    """Replacing a word with itself is unobservable - a stand desync, not an answer property."""
    _, text = _handmade()
    table = RuleTable(
        rules=(Rule(source="tree", target="tree"), Rule(source="sun", target="rock")),
        vocab_id="v_test",
        seed_rules=7,
    )

    with pytest.raises(ContractViolationError) as error:
        build_reference(table, text)

    assert error.value.error_code == "REFERENCE_MARKUP_MISMATCH"
