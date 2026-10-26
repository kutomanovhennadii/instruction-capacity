"""Mirror tests of the contracts of a point validation result."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import (
    OUTCOME_CORRECT,
    OUTCOME_MISSING,
    OUTCOME_OTHER_WORD,
    OUTCOME_UNREPLACED,
    OUTCOME_WRONG_RULE_TARGET,
    OUTCOMES,
    RuleStat,
    TrialRecord,
    TrialResult,
)


def _make_record(
    t: int,
    *,
    rule_index: int = 1,
    observed: str | None = "cat",
    z: int = 1,
    outcome: str = OUTCOME_CORRECT,
) -> TrialRecord:
    """Assemble the record of one trial with a correct outcome by default."""
    return TrialRecord(
        t=t,
        rule_index=rule_index,
        word_position=t * 2,
        expected="cat",
        observed=observed,
        z=z,
        outcome=outcome,
    )


def _make_result(
    records: tuple[TrialRecord, ...],
    *,
    rule_stats: tuple[RuleStat, ...] = (),
    first_error_t: int | None = None,
    first_error_rule_index: int | None = None,
    first_fail_kind_2_line: int | None = None,
    first_fail_kind_3_line: int | None = None,
    broken_line_count: int = 0,
    background_errors: int = 0,
    p_strict: bool = True,
    wrapper_leading_lines: int = 0,
    wrapper_trailing_lines: int = 0,
    wrapper_stripped: bool = False,
) -> TrialResult:
    """Assemble a point summary: by default error-free, without collapses."""
    return TrialResult(
        records=records,
        rule_stats=rule_stats,
        first_error_t=first_error_t,
        first_error_rule_index=first_error_rule_index,
        first_fail_kind_2_line=first_fail_kind_2_line,
        first_fail_kind_3_line=first_fail_kind_3_line,
        wrapper_leading_lines=wrapper_leading_lines,
        wrapper_trailing_lines=wrapper_trailing_lines,
        wrapper_stripped=wrapper_stripped,
        broken_line_count=broken_line_count,
        background_errors=background_errors,
        p_strict=p_strict,
    )


def test_outcomes_enumerate_all_categories_of_the_protocol() -> None:
    assert OUTCOMES == (
        OUTCOME_CORRECT,
        OUTCOME_UNREPLACED,
        OUTCOME_WRONG_RULE_TARGET,
        OUTCOME_OTHER_WORD,
        OUTCOME_MISSING,
    )
    assert len(set(OUTCOMES)) == len(OUTCOMES)


def test_record_fields_are_stored_as_given() -> None:
    record = TrialRecord(
        t=7,
        rule_index=3,
        word_position=41,
        expected="cat",
        observed="stone",
        z=0,
        outcome=OUTCOME_OTHER_WORD,
    )

    assert record.t == 7
    assert record.rule_index == 3
    assert record.word_position == 41
    assert record.expected == "cat"
    assert record.observed == "stone"
    assert record.z == 0
    assert record.outcome == OUTCOME_OTHER_WORD


def test_record_observed_is_none_when_position_unreachable() -> None:
    record = _make_record(4, observed=None, z=0, outcome=OUTCOME_MISSING)

    assert record.observed is None
    assert record.z == 0


def test_record_as_dict_is_json_serializable_round_trip() -> None:
    record = _make_record(2, rule_index=5, observed=None, z=0, outcome=OUTCOME_MISSING)

    snapshot = record.as_dict()

    assert snapshot == {
        "t": 2,
        "rule_index": 5,
        "word_position": 4,
        "expected": "cat",
        "observed": None,
        "z": 0,
        "outcome": OUTCOME_MISSING,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_record_is_frozen() -> None:
    record = _make_record(1)

    with pytest.raises(FrozenInstanceError):
        record.z = 0  # type: ignore[misc]


def test_rule_stat_p_rule_is_derived_from_counts() -> None:
    stat = RuleStat(rule_index=2, attempts=4, correct=1)

    assert stat.p_rule == 0.25


def test_rule_stat_p_rule_is_zero_without_attempts() -> None:
    stat = RuleStat(rule_index=2, attempts=0, correct=0)

    assert stat.p_rule == 0.0


def test_rule_stat_as_dict_is_json_serializable_round_trip() -> None:
    stat = RuleStat(rule_index=2, attempts=4, correct=3)

    snapshot = stat.as_dict()

    assert snapshot == {
        "rule_index": 2,
        "attempts": 4,
        "correct": 3,
        "p_rule": 0.75,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_rule_stat_is_frozen() -> None:
    stat = RuleStat(rule_index=1, attempts=2, correct=2)

    with pytest.raises(FrozenInstanceError):
        stat.correct = 0  # type: ignore[misc]


def test_result_fields_are_stored_as_given() -> None:
    records = (
        _make_record(1),
        _make_record(2, observed="tree", z=0, outcome=OUTCOME_UNREPLACED),
    )
    stats = (RuleStat(rule_index=1, attempts=2, correct=1),)
    result = _make_result(
        records,
        rule_stats=stats,
        first_error_t=2,
        first_error_rule_index=1,
        first_fail_kind_2_line=17,
        first_fail_kind_3_line=105,
        broken_line_count=3,
        background_errors=4,
        p_strict=False,
    )

    assert result.records == records
    assert result.rule_stats == stats
    assert result.first_error_t == 2
    assert result.first_error_rule_index == 1
    assert result.first_fail_kind_2_line == 17
    assert result.first_fail_kind_3_line == 105
    assert result.broken_line_count == 3
    assert result.background_errors == 4
    assert result.p_strict is False


def test_result_aggregates_are_derived_from_records() -> None:
    records = (
        _make_record(1),
        _make_record(2, observed="tree", z=0, outcome=OUTCOME_UNREPLACED),
        _make_record(3, observed="dog", z=0, outcome=OUTCOME_WRONG_RULE_TARGET),
        _make_record(4),
    )
    result = _make_result(records)

    assert result.h == 4
    assert result.correct_count == 2
    assert result.p_event == 0.5
    assert result.err_rate == 0.5


def test_result_flawless_run_has_full_p_event() -> None:
    result = _make_result((_make_record(1), _make_record(2)))

    assert result.p_event == 1.0
    assert result.err_rate == 0.0


def test_result_aggregates_are_zero_without_records() -> None:
    result = _make_result(())

    assert result.h == 0
    assert result.correct_count == 0
    assert result.p_event == 0.0
    assert result.err_rate == 1.0


def test_result_absence_of_failures_is_none_not_zero() -> None:
    """A censored observation: "no errors" is distinguishable from "an error at the end"."""
    result = _make_result((_make_record(1), _make_record(2)))

    assert result.first_error_t is None
    assert result.first_error_rule_index is None
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None


def test_result_as_dict_is_json_serializable_round_trip() -> None:
    result = _make_result(
        (
            _make_record(1),
            _make_record(2, observed="tree", z=0, outcome=OUTCOME_UNREPLACED),
        ),
        rule_stats=(RuleStat(rule_index=1, attempts=2, correct=1),),
        first_error_t=2,
        first_error_rule_index=1,
        first_fail_kind_2_line=1,
        broken_line_count=1,
        background_errors=2,
        p_strict=False,
    )

    snapshot = result.as_dict()

    assert snapshot == {
        "records": [
            {
                "t": 1,
                "rule_index": 1,
                "word_position": 2,
                "expected": "cat",
                "observed": "cat",
                "z": 1,
                "outcome": OUTCOME_CORRECT,
            },
            {
                "t": 2,
                "rule_index": 1,
                "word_position": 4,
                "expected": "cat",
                "observed": "tree",
                "z": 0,
                "outcome": OUTCOME_UNREPLACED,
            },
        ],
        "rule_stats": [
            {"rule_index": 1, "attempts": 2, "correct": 1, "p_rule": 0.5},
        ],
        "h": 2,
        "correct_count": 1,
        "p_event": 0.5,
        "err_rate": 0.5,
        "first_error_t": 2,
        "first_error_rule_index": 1,
        "first_fail_kind_2_line": 1,
        "first_fail_kind_3_line": None,
        "broken_line_count": 1,
        "background_errors": 2,
        "p_strict": False,
        "wrapper_leading_lines": 0,
        "wrapper_trailing_lines": 0,
        "wrapper_stripped": False,
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_wrapper_fields_default_to_no_wrapper() -> None:
    """Compatible extension: old constructors work, there is no wrapper by default."""
    result = _make_result((_make_record(1),))

    assert result.wrapper_leading_lines == 0
    assert result.wrapper_trailing_lines == 0
    assert result.wrapper_stripped is False


def test_wrapper_fields_are_carried_into_the_snapshot() -> None:
    """A violation of the answer form is recorded separately from the trial verdicts."""
    result = _make_result(
        (_make_record(1),),
        wrapper_leading_lines=2,
        wrapper_trailing_lines=1,
        wrapper_stripped=True,
    )

    snapshot = result.as_dict()

    assert snapshot["wrapper_leading_lines"] == 2
    assert snapshot["wrapper_trailing_lines"] == 1
    assert snapshot["wrapper_stripped"] is True
    assert result.p_event == 1.0


def test_result_is_frozen() -> None:
    result = _make_result((_make_record(1),))

    with pytest.raises(FrozenInstanceError):
        result.background_errors = 9  # type: ignore[misc]
