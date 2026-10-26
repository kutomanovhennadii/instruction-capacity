"""Normalized technical failures of the reliable call layer.

The transport and the provider adapter classify a failure each at its own level and name
it from a closed list instead of propagating an exception of a foreign library. These
classes do not decide the fate of the call (whether to retry, how long to wait) - that is
the business of ``ReliableModelClient``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from iec.infrastructure.exceptions import ProviderError


class TransportFailureKind(StrEnum):
    CONNECT_TIMEOUT = "connect_timeout"
    READ_TIMEOUT = "read_timeout"
    WRITE_TIMEOUT = "write_timeout"
    CONNECT_FAILURE = "connect_failure"
    CONNECTION_RESET = "connection_reset"
    REMOTE_PROTOCOL_FAILURE = "remote_protocol_failure"
    UNKNOWN_TRANSPORT_FAILURE = "unknown_transport_failure"


class ProviderFailureKind(StrEnum):
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
    retryable: bool
    retry_after_s: float | None = None


class TransportFailure(ProviderError):
    """An HTTP request was not delivered: the name did not resolve, the connection was not
    established, the link broke or the time ran out. A response with any HTTP code is not
    this error."""

    def __init__(self, message: str, *, kind: TransportFailureKind, elapsed_s: float) -> None:
        super().__init__(message, error_code=f"TRANSPORT_{kind.name}")
        self.kind = kind
        self.elapsed_s = elapsed_s


class ProviderFailure(ProviderError):
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
        super().__init__(message, error_code=f"PROVIDER_{kind.name}")
        self.provider = provider
        self.kind = kind
        self.http_status = http_status
        self.retry_hint = retry_hint
        self.provider_code = provider_code


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
        if category is ModelFailureCategory.TIMEOUT:
            error_code = "PROVIDER_TIMEOUT"
        elif category is ModelFailureCategory.INVALID_RESPONSE:
            error_code = "PROVIDER_RESPONSE_INVALID"
        else:
            error_code = "PROVIDER_API_ERROR"
        super().__init__(message, error_code=error_code)
        self.category = category
        self.attempts = attempts
        self.provider = provider
        self.model_id = model_id
