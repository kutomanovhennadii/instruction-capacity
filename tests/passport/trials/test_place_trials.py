"""Trials of P_sync, H and F by places: outcomes, the word number of an insertion, deletions."""

from __future__ import annotations

import pytest

from iec.answer_validator import align_answer, build_reference
from iec.contracts import (
    PointConditions,
    PointRecord,
    PointStatus,
    Reference,
    RuleTable,
    SourceText,
)
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.place_trials import PointTrials, point_trials
from iec.passport.trials.trial_tables import build_trial_tables
from iec.point_execution import task_from_record

RULES = (("lake", "stone"), ("tree", "fish"))


def _source_line(index: int) -> list[str]:
    words = [f"{'abcdefgh'[index]}{'abcdefghij'[place]}q" for place in range(10)]
    words[index % 10] = "lake"
    words[(index + 5) % 10] = "tree"
    return words


SOURCE = [_source_line(index) for index in range(8)]
REFERENCE = [
    ["stone" if word == "lake" else "fish" if word == "tree" else word for word in line]
    for line in SOURCE
]


def _task() -> tuple[RuleTable, SourceText, Reference]:
    record = PointRecord(
        conditions=PointConditions(D=66, n=1, m=2, seed_rules=1, seed_text=2, seed_order=0),
        rules=RULES,
        source_text="\n".join(" ".join(line) for line in SOURCE),
        answer="",
        status=PointStatus.OK,
        prompt_tokens=None,
        response_tokens=None,
    )
    table, text = task_from_record(record, "v1_intersection")
    return table, text, build_reference(table, text)


def _lines(*numbers: int) -> list[list[str]]:
    return [list(REFERENCE[number]) for number in numbers]


def _trials(lines: list[list[str]]) -> PointTrials:
    table, text, reference = _task()
    aligned = align_answer("\n".join(" ".join(line) for line in lines), reference, text, table)
    tables = build_trial_tables(len(reference.words))
    return point_trials("k", 66, 1, 2, aligned, reference, text, table, tables=tables)


def _sync(trials: PointTrials) -> list[tuple[int, int, int]]:
    return [(s_bin, places, failures) for s_bin, places, failures, _ in trials.sync]


def test_answer_equal_to_reference_has_no_failures() -> None:
    trials = _trials(_lines(*range(8)))

    assert _sync(trials) == [(1, 10, 0), (2, 10, 0), (3, 20, 0), (4, 40, 0)]
    assert (trials.totals["rule_trials"], trials.totals["rule_correct"]) == (16, 16)
    assert (trials.totals["background_trials"], trials.totals["background_misfires"]) == (64, 0)
    assert sum(failures for _, _, _, failures, _ in trials.rule) == 0


def test_outcomes_of_rule_and_background_words() -> None:
    lines = _lines(*range(8))
    lines[2][1] = "fish"  # a misfire: line 3, a background word
    lines[5][5] = "fish"  # another rule: line 6, the place of lake
    lines[3][3] = "lake"  # the rule is not applied: line 4, the place of lake

    trials = _trials(lines)

    totals = trials.totals
    assert (totals["rule_correct"], totals["rule_wrong_rule"], totals["rule_unreplaced"]) == (
        14,
        1,
        1,
    )
    assert (totals["background_misfires"], totals["background_other_change"]) == (1, 0)
    rule = {(s_bin, index): (count, failures) for s_bin, index, count, failures, _ in trials.rule}
    assert rule[(3, 1)] == (2, 1)
    assert rule[(4, 1)] == (4, 1)
    background = {s_bin: (count, failures) for s_bin, count, failures, _ in trials.background}
    assert background[3] == (16, 1)


def test_inserted_repeat_stands_before_the_next_reference_word() -> None:
    trials = _trials(_lines(0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7))

    # The repeat of lines 2-4 stands before word 41 (line 5, bin 4): 40 places and 30 insertions.
    assert _sync(trials) == [(1, 10, 0), (2, 10, 0), (3, 20, 0), (4, 70, 30)]
    assert trials.totals["inserted"] == 30


def test_skipped_lines_are_deleted_rule_and_background_words() -> None:
    trials = _trials(_lines(0, 1, 4, 5, 6, 7))

    assert (trials.totals["rule_deleted"], trials.totals["background_deleted"]) == (4, 16)
    assert _sync(trials)[2] == (3, 20, 20)


def test_tables_shorter_than_the_text_are_an_error() -> None:
    table, text, reference = _task()
    answer = "\n".join(" ".join(line) for line in _lines(*range(8)))
    aligned = align_answer(answer, reference, text, table)

    with pytest.raises(ContractViolationError) as caught:
        point_trials("k", 66, 1, 2, aligned, reference, text, table, tables=build_trial_tables(79))

    assert caught.value.error_code == "TRIAL_TABLES_INVALID"
