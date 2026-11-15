"""Mirror tests of the normalized failures and of the policy of every kind."""

import pytest

from iec.model_client.reliability.failures import (
    ModelCallFailure,
    ModelFailureCategory,
    ProviderFailure,
    ProviderFailureKind,
    RetryHint,
    TransportFailure,
    TransportFailureKind,
)

Category = ModelFailureCategory
Transport = TransportFailureKind
Provider = ProviderFailureKind

#: Expected treatment of every transport kind: category, retryable, counts against the circuit,
#: timeout.
_TRANSPORT = {
    Transport.CONNECT_TIMEOUT: (Category.TIMEOUT, False, True, True),
    Transport.READ_TIMEOUT: (Category.TIMEOUT, False, False, True),
    Transport.WRITE_TIMEOUT: (Category.TIMEOUT, False, False, True),
    Transport.CONNECT_FAILURE: (Category.NETWORK_UNAVAILABLE, True, True, False),
    Transport.CONNECTION_RESET: (Category.NETWORK_UNAVAILABLE, True, True, False),
    Transport.REMOTE_PROTOCOL_FAILURE: (Category.NETWORK_UNAVAILABLE, True, True, False),
    Transport.UNKNOWN_TRANSPORT_FAILURE: (Category.NETWORK_UNAVAILABLE, True, True, False),
}

#: Expected treatment of every provider kind: category, retryable when the hint asks for a retry,
#: counts against the circuit.
_PROVIDER = {
    Provider.RATE_LIMITED: (Category.RATE_LIMIT, True, False),
    Provider.QUOTA_EXHAUSTED: (Category.QUOTA_EXHAUSTED, True, False),
    Provider.AUTHENTICATION_FAILED: (Category.AUTH_FAILURE, True, False),
    Provider.PERMISSION_DENIED: (Category.AUTH_FAILURE, True, False),
    Provider.MODEL_NOT_FOUND: (Category.INVALID_REQUEST, True, False),
    Provider.PROVIDER_OVERLOADED: (Category.PROVIDER_UNAVAILABLE, True, True),
    Provider.INVALID_REQUEST: (Category.INVALID_REQUEST, True, False),
    Provider.INVALID_RESPONSE: (Category.INVALID_RESPONSE, False, False),
    Provider.UNKNOWN_PROVIDER_FAILURE: (Category.PROVIDER_UNAVAILABLE, True, False),
}


@pytest.mark.parametrize("kind", list(Transport))
def test_a_transport_failure_follows_the_policy_of_its_kind(kind: TransportFailureKind) -> None:
    category, retryable, counts, is_timeout = _TRANSPORT[kind]

    failure = TransportFailure("failed", kind=kind, elapsed_s=0.1)

    assert failure.policy.category is category
    assert failure.retryable is retryable
    assert failure.policy.counts_against_circuit is counts
    assert failure.policy.is_timeout is is_timeout
    assert failure.retry_after_s is None


@pytest.mark.parametrize("kind", list(Provider))
def test_a_provider_failure_follows_its_kind_and_its_hint(kind: ProviderFailureKind) -> None:
    category, retryable, counts = _PROVIDER[kind]
    hint = RetryHint(retryable=True)

    hinted = ProviderFailure("failed", provider="fake", kind=kind, retry_hint=hint)
    unhinted = ProviderFailure("failed", provider="fake", kind=kind)

    assert hinted.policy.category is category
    assert hinted.policy.counts_against_circuit is counts
    assert hinted.policy.is_timeout is False
    assert hinted.retryable is retryable
    assert unhinted.retryable is False


def test_a_provider_failure_keeps_the_pause_the_provider_asked_for() -> None:
    failure = ProviderFailure(
        "rate limited",
        provider="fake",
        kind=Provider.RATE_LIMITED,
        retry_hint=RetryHint(retryable=True, retry_after_s=7.0),
    )

    assert failure.retry_after_s == 7.0


@pytest.mark.parametrize("category", list(Category))
def test_the_final_failure_code_follows_its_category(category: ModelFailureCategory) -> None:
    expected = {
        Category.TIMEOUT: "PROVIDER_TIMEOUT",
        Category.INVALID_RESPONSE: "PROVIDER_RESPONSE_INVALID",
    }.get(category, "PROVIDER_API_ERROR")

    failure = ModelCallFailure(
        "failed", category=category, attempts=1, provider="fake", model_id="model"
    )

    assert failure.error_code == expected
