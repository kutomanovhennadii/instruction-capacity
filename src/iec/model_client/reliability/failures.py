"""Normalized technical failures of the reliable call layer and the policy of every kind.

The transport and the provider adapter classify a failure each at its own level and name it
from a closed list instead of propagating an exception of a foreign library. How the reliable
call layer treats a failure - whether a retry may cure it, whether it speaks against the health
of the provider, which category the final failure gets - is the policy of its kind, written in
the tables below. ``ReliableModelClient`` reads the policy through the common interface
``ClassifiedFailure`` and never checks the class of a failure (specification §2.10).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from iec.infrastructure.exceptions import ProviderError


class TransportFailureKind(StrEnum):
    """Why an HTTP request was not delivered."""

    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    WRITE_TIMEOUT = "write_timeout"
    CONNECT_FAILURE = "connect_failure"
    CONNECTION_RESET = "connection_reset"
    REMOTE_PROTOCOL_FAILURE = "remote_protocol_failure"
    UNKNOWN_TRANSPORT_FAILURE = "unknown_transport_failure"


class ProviderFailureKind(StrEnum):
    """What went wrong in an error response of a provider, as its adapter classifies it."""

    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    MODEL_NOT_FOUND = "model_not_found"
    PROVIDER_OVERLOADED = "provider_overloaded"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN_PROVIDER_FAILURE = "unknown_provider_failure"


class ModelFailureCategory(StrEnum):
    """Category of the final failure of a model call, as seen above the reliable layer."""

    TIMEOUT = "timeout"
    NETWORK_UNAVAILABLE = "network_unavailable"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXHAUSTED = "quota_exhausted"
    AUTH_FAILURE = "auth_failure"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    INVALID_RESPONSE = "invalid_response"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RetryHint:
    """The opinion of an adapter about a retry and the pause the provider asked for."""

    retryable: bool
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    """How the reliable call layer treats a failure of one kind.

    ``category`` is the category of the final failure of the call. ``retry_allowed`` tells
    whether the kind allows a retry; a provider failure needs the hint of its adapter too, and a
    timeout follows ``TIMEOUT_RETRIES`` instead. ``counts_against_circuit`` tells whether the
    failure speaks against the health of the provider and moves the circuit breaker.
    ``is_timeout`` tells whether the time ran out: such a failure is retried without a pause.
    """

    category: ModelFailureCategory
    retry_allowed: bool = True
    counts_against_circuit: bool = False
    is_timeout: bool = False


#: Policies of the transport failures. A failure to deliver a request is retried and speaks
#: against the channel. Only a timeout of the connection counts against the circuit: the
#: handshake does not depend on the weight of the task, and its failure speaks about the channel.
#: A read or write timeout means that the model did not fit into the limit on this node - a
#: property of the node, not of the provider: a batch at the edge of the measured area can hold
#: several heavy nodes in a row, and a breaker opened by their silence would hand out failures to
#: the neighbouring points of the same batch.
TRANSPORT_POLICIES: Mapping[TransportFailureKind, FailurePolicy] = {
    TransportFailureKind.CONNECT_TIMEOUT: FailurePolicy(
        ModelFailureCategory.TIMEOUT,
        retry_allowed=False,
        counts_against_circuit=True,
        is_timeout=True,
    ),
    TransportFailureKind.READ_TIMEOUT: FailurePolicy(
        ModelFailureCategory.TIMEOUT, retry_allowed=False, is_timeout=True
    ),
    TransportFailureKind.WRITE_TIMEOUT: FailurePolicy(
        ModelFailureCategory.TIMEOUT, retry_allowed=False, is_timeout=True
    ),
    TransportFailureKind.CONNECT_FAILURE: FailurePolicy(
        ModelFailureCategory.NETWORK_UNAVAILABLE, counts_against_circuit=True
    ),
    TransportFailureKind.CONNECTION_RESET: FailurePolicy(
        ModelFailureCategory.NETWORK_UNAVAILABLE, counts_against_circuit=True
    ),
    TransportFailureKind.REMOTE_PROTOCOL_FAILURE: FailurePolicy(
        ModelFailureCategory.NETWORK_UNAVAILABLE, counts_against_circuit=True
    ),
    TransportFailureKind.UNKNOWN_TRANSPORT_FAILURE: FailurePolicy(
        ModelFailureCategory.NETWORK_UNAVAILABLE, counts_against_circuit=True
    ),
}

#: Policies of the provider failures. A retry of a provider failure also needs the hint of its
#: adapter: lack of credits comes with the same status 429 as a rate limit, and only the adapter
#: tells them apart. A broken response body is never retried: a second attempt gets the same body.
#: Only an overload of the provider (statuses 500-504) counts against the circuit.
PROVIDER_POLICIES: Mapping[ProviderFailureKind, FailurePolicy] = {
    ProviderFailureKind.RATE_LIMITED: FailurePolicy(ModelFailureCategory.RATE_LIMIT),
    ProviderFailureKind.QUOTA_EXHAUSTED: FailurePolicy(ModelFailureCategory.QUOTA_EXHAUSTED),
    ProviderFailureKind.AUTHENTICATION_FAILED: FailurePolicy(ModelFailureCategory.AUTH_FAILURE),
    ProviderFailureKind.PERMISSION_DENIED: FailurePolicy(ModelFailureCategory.AUTH_FAILURE),
    ProviderFailureKind.MODEL_NOT_FOUND: FailurePolicy(ModelFailureCategory.INVALID_REQUEST),
    ProviderFailureKind.PROVIDER_OVERLOADED: FailurePolicy(
        ModelFailureCategory.PROVIDER_UNAVAILABLE, counts_against_circuit=True
    ),
    ProviderFailureKind.INVALID_REQUEST: FailurePolicy(ModelFailureCategory.INVALID_REQUEST),
    ProviderFailureKind.INVALID_RESPONSE: FailurePolicy(
        ModelFailureCategory.INVALID_RESPONSE, retry_allowed=False
    ),
    ProviderFailureKind.UNKNOWN_PROVIDER_FAILURE: FailurePolicy(
        ModelFailureCategory.PROVIDER_UNAVAILABLE
    ),
}

#: Error code of the final failure of a model call by its category; every other category gives
#: ``PROVIDER_API_ERROR``.
_CATEGORY_ERROR_CODES: Mapping[ModelFailureCategory, str] = {
    ModelFailureCategory.TIMEOUT: "PROVIDER_TIMEOUT",
    ModelFailureCategory.INVALID_RESPONSE: "PROVIDER_RESPONSE_INVALID",
}
_DEFAULT_CALL_ERROR_CODE = "PROVIDER_API_ERROR"


class ClassifiedFailure(ProviderError, ABC):
    """A failure of one attempt of a model call, named from a closed list of kinds.

    The reliable layer reads its treatment from :attr:`policy` and never checks the class of the
    failure: a new kind of failure is a new entry in a policy table.
    """

    @property
    @abstractmethod
    def policy(self) -> FailurePolicy:
        """Return the policy of the kind of this failure."""

    @property
    def retryable(self) -> bool:
        """Tell whether a retry may cure this failure; a timeout follows ``TIMEOUT_RETRIES``."""
        return self.policy.retry_allowed

    @property
    def retry_after_s(self) -> float | None:
        """Return the pause the provider asked for before a retry, or ``None``."""
        return None


class TransportFailure(ClassifiedFailure):
    """An HTTP request was not delivered: the name did not resolve, the connection was not
    established, the link broke or the time ran out. A response with any HTTP code is not
    this error."""

    def __init__(self, message: str, *, kind: TransportFailureKind, elapsed_s: float) -> None:
        """Name the failure by its kind.

        Args:
            message: Text of the failure for a human.
            kind: Why the request was not delivered; the error code is ``TRANSPORT_<KIND>``.
            elapsed_s: Seconds from the start of the request to the failure.
        """
        super().__init__(message, error_code=f"TRANSPORT_{kind.name}")
        self.kind = kind
        self.elapsed_s = elapsed_s

    @property
    def policy(self) -> FailurePolicy:
        """Return the policy of the kind of this transport failure."""
        return TRANSPORT_POLICIES[self.kind]


class ProviderFailure(ClassifiedFailure):
    """A provider response with an error (HTTP status >= 400 or an unparsable body),
    classified by the adapter of a particular provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        kind: ProviderFailureKind,
        http_status: int | None = None,
        retry_hint: RetryHint = RetryHint(retryable=False),
        provider_code: str | None = None,
    ) -> None:
        """Name the failure by its kind and keep what the adapter learned from the response.

        Args:
            message: Text of the failure for a human.
            provider: Identifier of the provider.
            kind: What went wrong; the error code is ``PROVIDER_<KIND>``.
            http_status: Status of the response, when there was a response.
            retry_hint: Whether a retry may help and the pause the provider asked for.
            provider_code: Error code of the provider from the response body.
        """
        super().__init__(message, error_code=f"PROVIDER_{kind.name}")
        self.provider = provider
        self.kind = kind
        self.http_status = http_status
        self.retry_hint = retry_hint
        self.provider_code = provider_code

    @property
    def policy(self) -> FailurePolicy:
        """Return the policy of the kind of this provider failure."""
        return PROVIDER_POLICIES[self.kind]

    @property
    def retryable(self) -> bool:
        """Tell whether a retry may cure this failure: the kind allows it and the hint asks."""
        return self.policy.retry_allowed and self.retry_hint.retryable

    @property
    def retry_after_s(self) -> float | None:
        """Return the pause the provider asked for in ``Retry-After``, or ``None``."""
        return self.retry_hint.retry_after_s


class ModelCallFailure(ProviderError):
    """The final failure of a model call as seen by the upper boundary
    (``OpenAIModelClient.generate`` and its analogues for other providers)."""

    def __init__(
        self,
        message: str,
        *,
        category: ModelFailureCategory,
        attempts: int,
        provider: str,
        model_id: str,
    ) -> None:
        """Keep the category and the facts of the call; the error code follows the category.

        Args:
            message: Text of the failure for a human.
            category: Category of the final failure.
            attempts: Number of attempts made.
            provider: Identifier of the provider.
            model_id: Identifier of the model.
        """
        error_code = _CATEGORY_ERROR_CODES.get(category, _DEFAULT_CALL_ERROR_CODE)
        super().__init__(message, error_code=error_code)
        self.category = category
        self.attempts = attempts
        self.provider = provider
        self.model_id = model_id
