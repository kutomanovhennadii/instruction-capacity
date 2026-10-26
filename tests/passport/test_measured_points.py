"""Published records become measured points: the validator rebuilds the trials."""

from pathlib import Path

import pytest

from iec.answer_validator import build_reference
from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError
from iec.model_client import MockTokenizer
from iec.passport.measured_points import measure_points
from iec.prompt_generator import RuleGenerator, TextGenerator
from iec.vocab import VocabStore

_VOCAB_DIR = Path(__file__).resolve().parents[2] / "assets" / "vocab"
_CONDITIONS = PointConditions(D=8, n=5, m=2, seed_rules=11, seed_text=12, seed_order=0)


def _record(status: PointStatus, answer: str | None) -> PointRecord:
    vocab = VocabStore(_VOCAB_DIR).load("v1_intersection")
    table = RuleGenerator(vocab).generate(_CONDITIONS)
    text = TextGenerator(vocab, MockTokenizer()).generate(_CONDITIONS, table)
    if answer == "reference":
        answer = build_reference(table, text).text
    return PointRecord(
        conditions=_CONDITIONS,
        rules=tuple((rule.source, rule.target) for rule in table.rules),
        source_text=text.text,
        answer=answer,
        status=status,
        prompt_tokens=None,
        response_tokens=None,
    )


def test_a_right_answer_gives_one_right_trial_per_replaced_word() -> None:
    (point,) = measure_points([_record(PointStatus.OK, "reference")], "v1_intersection")
    # Two rules over eight distinct words, each word five times: ten replaced words.
    assert len(point.trials) == 10
    assert all(trial.z == 1 for trial in point.trials)
    assert point.first_fail_kind_3_line is None
    assert point.key == "8/5/2/11/12/0"
    assert (point.D, point.n, point.m, point.status) == (8, 5, 2, PointStatus.OK)


def test_an_empty_answer_loses_the_text_from_the_first_line() -> None:
    (point,) = measure_points([_record(PointStatus.OK, "")], "v1_intersection")
    assert len(point.trials) == 10
    assert all(trial.z == 0 for trial in point.trials)
    assert point.first_fail_kind_3_line == 1


def test_a_point_without_a_call_has_no_trials() -> None:
    (point,) = measure_points([_record(PointStatus.NA, None)], "v1_intersection")
    assert point.trials == ()
    assert point.first_fail_kind_3_line is None


def test_an_ok_record_without_an_answer_is_refused() -> None:
    with pytest.raises(ContractViolationError) as error:
        measure_points([_record(PointStatus.OK, None)], "v1_intersection")
    assert error.value.error_code == "POINT_RECORD_INVALID"
