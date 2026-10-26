import json

import pytest

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.infrastructure.exceptions import ContractViolationError

_CONDITIONS = PointConditions(D=4, n=5, m=1, seed_rules=10**38, seed_text=7, seed_order=0)


def _record(**changes: object) -> PointRecord:
    fields: dict[str, object] = {
        "conditions": _CONDITIONS,
        "rules": (("lake", "stone"),),
        "source_text": "lake a b c d e f g h i\nlake a b c d e f g h i",
        "answer": "stone a b c d e f g h i\nstone a b c d e f g h i",
        "status": PointStatus.OK,
        "prompt_tokens": 107,
        "response_tokens": 26,
    }
    fields.update(changes)
    return PointRecord(**fields)  # type: ignore[arg-type]


def test_record_survives_the_json_round_trip_with_big_seeds() -> None:
    record = _record()
    line = json.dumps(record.as_dict())
    assert PointRecord.from_dict(json.loads(line)) == record


def test_published_keys_and_order() -> None:
    assert list(_record().as_dict()) == [
        "D", "n", "m", "seed_rules", "seed_text", "seed_order",
        "rules", "source_text", "answer", "status", "prompt_tokens", "response_tokens",
    ]


def test_unknown_keys_are_ignored() -> None:
    raw = {**_record().as_dict(), "phase": "grid"}
    assert PointRecord.from_dict(raw) == _record()


def test_na_record_keeps_the_task_without_an_answer() -> None:
    record = _record(answer=None, status=PointStatus.NA, prompt_tokens=None, response_tokens=None)
    assert PointRecord.from_dict(record.as_dict()) == record


@pytest.mark.parametrize("broken", [{"status": "stand_error"}, {"seed_order": None}])
def test_broken_record_raises_contract_violation(broken: dict[str, object]) -> None:
    raw = {**_record().as_dict(), **broken}
    with pytest.raises(ContractViolationError) as excinfo:
        PointRecord.from_dict(raw)
    assert excinfo.value.error_code == "POINT_RECORD_INVALID"


def test_conditions_identity_holds_all_six_numbers() -> None:
    assert _CONDITIONS.identity == (4, 5, 1, 10**38, 7, 0)
    assert _CONDITIONS.v == 20
