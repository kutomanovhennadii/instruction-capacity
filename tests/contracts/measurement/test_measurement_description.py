"""Mirror tests of the description of a published measurement and of its format version."""

import pytest

from iec.contracts import (
    MEASUREMENT_DESCRIPTION_FORMAT_VERSION,
    GenParams,
    MeasurementDescription,
    ModelConfig,
    Tariff,
)
from iec.contracts.format_version import FORMAT_VERSION_UNSUPPORTED
from iec.contracts.measurement.measurement_description import MEASUREMENT_DESCRIPTION_INVALID
from iec.infrastructure.exceptions import ContractViolationError

_DESCRIPTION = MeasurementDescription(
    model=ModelConfig(model_id="m", provider="mock", gen_params=GenParams()),
    protocol_version="0.1",
    vocab_id="v1_intersection",
    vocab_hash="sha256:abc",
    tariff=Tariff(input_usd_per_million=0.15, output_usd_per_million=0.6),
    files={"points.jsonl.gz": "00ff"},
)


def test_measurement_description_round_trip() -> None:
    assert MeasurementDescription.from_dict(_DESCRIPTION.as_dict()) == _DESCRIPTION


def test_dict_carries_the_format_version_first() -> None:
    raw = _DESCRIPTION.as_dict()

    assert next(iter(raw)) == "format_version"
    assert raw["format_version"] == MEASUREMENT_DESCRIPTION_FORMAT_VERSION == 1


def test_description_without_a_version_reads_as_version_one() -> None:
    """The published measurements, v55 among them, have no format_version."""
    raw = _DESCRIPTION.as_dict()
    del raw["format_version"]

    assert MeasurementDescription.from_dict(raw) == _DESCRIPTION


def test_newer_format_is_refused() -> None:
    raw = {**_DESCRIPTION.as_dict(), "format_version": MEASUREMENT_DESCRIPTION_FORMAT_VERSION + 1}

    with pytest.raises(ContractViolationError) as caught:
        MeasurementDescription.from_dict(raw)

    assert caught.value.error_code == FORMAT_VERSION_UNSUPPORTED


def test_bad_format_version_is_invalid() -> None:
    raw = {**_DESCRIPTION.as_dict(), "format_version": "1"}

    with pytest.raises(ContractViolationError) as caught:
        MeasurementDescription.from_dict(raw)

    assert caught.value.error_code == MEASUREMENT_DESCRIPTION_INVALID


def test_missing_key_is_invalid_and_keeps_its_cause() -> None:
    raw = _DESCRIPTION.as_dict()
    del raw["vocab_id"]

    with pytest.raises(ContractViolationError) as caught:
        MeasurementDescription.from_dict(raw)

    assert caught.value.error_code == MEASUREMENT_DESCRIPTION_INVALID
    assert isinstance(caught.value.__cause__, KeyError)


def test_files_are_written_sorted_by_name() -> None:
    description = MeasurementDescription(
        model=_DESCRIPTION.model,
        protocol_version="0.2",
        vocab_id="v",
        vocab_hash="sha256:0",
        files={"b.txt": "02", "a.txt": "01"},
    )

    assert list(description.as_dict()["files"]) == ["a.txt", "b.txt"]
