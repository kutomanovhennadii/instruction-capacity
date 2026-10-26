"""Mirror tests of the single IEC exception hierarchy."""

import pytest

from iec.infrastructure.exceptions import (
    ConfigError,
    ContractViolationError,
    IECError,
    ProviderError,
    ValidationError,
    VocabError,
)


@pytest.mark.parametrize(
    ("error_type", "expected_code"),
    [
        (IECError, "IEC_ERROR"),
        (ConfigError, "CONFIG_ERROR"),
        (VocabError, "VOCAB_ERROR"),
        (ProviderError, "PROVIDER_ERROR"),
        (ValidationError, "VALIDATION_ERROR"),
        (ContractViolationError, "CONTRACT_VIOLATION"),
    ],
)
def test_default_error_code(
    error_type: type[IECError],
    expected_code: str,
) -> None:
    error = error_type("test failure")

    assert str(error) == "test failure"
    assert error.message == "test failure"
    assert error.error_code == expected_code


def test_explicit_error_code_overrides_class_default() -> None:
    error = ConfigError("unknown provider", error_code="UNKNOWN_PROVIDER")

    assert error.error_code == "UNKNOWN_PROVIDER"


def test_from_cause_preserves_original_exception() -> None:
    cause = RuntimeError("provider unavailable")

    error = ProviderError.from_cause("model build failed", cause)

    assert isinstance(error, ProviderError)
    assert error.__cause__ is cause
    assert error.error_code == "PROVIDER_ERROR"


def test_from_cause_accepts_error_code_override() -> None:
    cause = ValueError("bad response")

    error = ProviderError.from_cause(
        "provider returned invalid data",
        cause,
        error_code="INVALID_PROVIDER_RESPONSE",
    )

    assert error.error_code == "INVALID_PROVIDER_RESPONSE"
    assert error.__cause__ is cause


def test_common_root_catches_subclass() -> None:
    error: IECError | None = None

    try:
        raise ValidationError("invalid output")
    except IECError as caught:
        error = caught

    assert isinstance(error, ValidationError)
