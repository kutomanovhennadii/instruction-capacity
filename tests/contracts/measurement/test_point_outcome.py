"""Mirror tests of the full result of executing one point."""

import dataclasses

import pytest

from iec.contracts import (
    PointConditions,
    PointOutcome,
    PointRecord,
    PointStatus,
    Prompt,
    Reference,
    Rule,
    RuleTable,
    SourceText,
)


def _outcome() -> PointOutcome:
    record = PointRecord(
        conditions=PointConditions(D=1, n=2, m=1, seed_rules=1, seed_text=2),
        rules=(("lake", "stone"),),
        source_text="lake lake",
        answer=None,
        status=PointStatus.NA,
        prompt_tokens=None,
        response_tokens=None,
    )
    return PointOutcome(
        record=record,
        rule_table=RuleTable(rules=(Rule("lake", "stone"),), vocab_id="v", seed_rules=1),
        source_text=SourceText(
            words=("lake", "lake"),
            words_per_line=2,
            replacement_positions=(0, 1),
            len_tokens=2,
            tokenizer_name="mock",
            seed_text=2,
        ),
        prompt=Prompt(text="Replace: lake -> stone"),
        reference=Reference(words=("stone", "stone"), words_per_line=2),
        prompt_len_tokens=5,
        raw_output=None,
        error_code=None,
        duration_s=0.0,
    )


def test_outcome_keeps_the_published_record_and_the_built_objects() -> None:
    outcome = _outcome()

    assert outcome.record.status is PointStatus.NA
    assert outcome.reference.text == "stone stone"
    assert outcome.raw_output is None


def test_outcome_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(_outcome(), "duration_s", 1.0)
