"""Mirror tests of the one rule of reading the format version of a stored contract."""

import pytest

from iec.contracts.format_version import FORMAT_VERSION_UNSUPPORTED, check_format_version
from iec.infrastructure.exceptions import ContractViolationError


def _check(raw: object) -> None:
    check_format_version(raw, 2, invalid_code="OBJECT_INVALID", what="an object")


@pytest.mark.parametrize("raw", [{}, {"format_version": 1}, {"format_version": 2}])
def test_versions_up_to_the_current_one_are_read(raw: dict[str, object]) -> None:
    """An object without the field is of version 1."""
    _check(raw)


def test_newer_version_is_unsupported() -> None:
    with pytest.raises(ContractViolationError) as caught:
        _check({"format_version": 3})

    assert caught.value.error_code == FORMAT_VERSION_UNSUPPORTED


@pytest.mark.parametrize("version", ["1", True, 0, 1.0, None])
def test_bad_version_raises_the_code_of_the_object(version: object) -> None:
    with pytest.raises(ContractViolationError) as caught:
        _check({"format_version": version})

    assert caught.value.error_code == "OBJECT_INVALID"


def test_object_that_is_not_a_json_object_raises_the_code_of_the_object() -> None:
    with pytest.raises(ContractViolationError) as caught:
        _check(["format_version", 1])

    assert caught.value.error_code == "OBJECT_INVALID"
