"""Resources of the published points and the alarm on incomplete token counters."""

import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus, Tariff, TrialRecord
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.measured_points import MeasuredPoint
from iec.passport.passport_data import INCOMPLETE_TOKEN_COUNTERS, PassportAlarm, _resources_block


def _point(
    status: PointStatus,
    prompt: int | None,
    response: int | None,
    *,
    trials: int = 0,
    seed: int = 1,
) -> MeasuredPoint:
    record = PointRecord(
        conditions=PointConditions(D=4, n=5, m=1, seed_rules=seed, seed_text=seed + 1, seed_order=0),
        rules=(),
        source_text="",
        answer="" if status is PointStatus.OK else None,
        status=status,
        prompt_tokens=prompt,
        response_tokens=response,
    )
    records = tuple(
        TrialRecord(
            t=index,
            rule_index=1,
            word_position=index,
            expected="a",
            observed="a",
            z=1,
            outcome="correct",
        )
        for index in range(1, trials + 1)
    )
    return MeasuredPoint(record=record, trials=records, first_fail_kind_3_line=None)


def test_resources_count_the_published_points_and_their_cost() -> None:
    points = [
        _point(PointStatus.OK, 100, 20, trials=3, seed=1),
        _point(PointStatus.OK, None, None, trials=2, seed=3),
        _point(PointStatus.API_ERROR, 50, 0, seed=5),
        _point(PointStatus.NA, None, None, seed=7),
    ]
    tariff = Tariff(input_usd_per_million=1.0, output_usd_per_million=2.0)
    resources, alarms = _resources_block(points, tariff)
    assert resources == {
        "attempted_calls": 3,
        "unique_point_keys": 4,
        "ok_runs": 2,
        "trials": 5,
        "prompt_tokens_billed": 150,
        "completion_tokens_billed": 20,
        "input_cost_usd": pytest.approx(150e-6),
        "output_cost_usd": pytest.approx(40e-6),
        "total_cost_usd": pytest.approx(190e-6),
    }
    assert [alarm.code for alarm in alarms] == [INCOMPLETE_TOKEN_COUNTERS]
    assert alarms[0].evidence == {"records": 1}


def test_without_a_tariff_the_money_is_not_computed_and_nothing_alarms() -> None:
    resources, alarms = _resources_block([_point(PointStatus.OK, 10, 5, trials=1)], None)
    assert resources["input_cost_usd"] is None
    assert resources["total_cost_usd"] is None
    assert alarms == []


def test_an_alarm_code_must_be_screaming_snake_case() -> None:
    with pytest.raises(ContractViolationError):
        PassportAlarm(severity="warning", code="lower_case", message="x")
