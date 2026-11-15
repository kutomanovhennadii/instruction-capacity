"""The transport layer: HTTP delivery only, without the semantics of any provider.

``HttpTransport`` can do one thing - send a request and return the response. A response
with code 429 or 500 is a normally received response for it, not an error: interpreting
HTTP statuses is the business of the provider adapter, not of the transport. The only
error here is non-delivery, and every such outcome is named from the closed list
``TransportFailureKind`` instead of propagating as an ``httpx`` exception. The kind is read
from a table by the classes of the exception, the most specific first.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from iec.model_client.reliability.failures import TransportFailure, TransportFailureKind


@dataclass(frozen=True, slots=True)
class HttpRequest:
    """An HTTP request as a provider adapter builds it: method, path, headers and body."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """A received HTTP response with any status code and the seconds it took."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_s: float


class CallContext(Protocol):
    """What the transport needs from the context of a call: the time left."""

    @property
    def remaining_s(self) -> float:
        """Return the seconds left until the overall deadline of the call."""
        ...


class HttpTransport(Protocol):
    """Delivery of one HTTP request; any received response is a result, not an error."""

    def send(self, request: HttpRequest, context: CallContext) -> HttpResponse:
        """Send the request and return the response.

        Raises:
            TransportFailure: the request was not delivered.
        """
        ...


#: Kinds of the transport failures by the class of the ``httpx`` exception. The classes of an
#: exception are looked up from the most specific one, so ``RemoteProtocolError`` takes the kind
#: of ``ProtocolError``; an exception of none of these classes is ``UNKNOWN_TRANSPORT_FAILURE``.
_FAILURE_KINDS: Mapping[type, TransportFailureKind] = {
    httpx.ConnectTimeout: TransportFailureKind.CONNECT_TIMEOUT,
    httpx.WriteTimeout: TransportFailureKind.WRITE_TIMEOUT,
    httpx.ReadTimeout: TransportFailureKind.READ_TIMEOUT,
    httpx.PoolTimeout: TransportFailureKind.READ_TIMEOUT,
    httpx.ConnectError: TransportFailureKind.CONNECT_FAILURE,
    httpx.ReadError: TransportFailureKind.CONNECTION_RESET,
    httpx.WriteError: TransportFailureKind.CONNECTION_RESET,
    httpx.CloseError: TransportFailureKind.CONNECTION_RESET,
    httpx.ProtocolError: TransportFailureKind.REMOTE_PROTOCOL_FAILURE,
}


def _failure_kind(error: httpx.TransportError) -> TransportFailureKind:
    """Return the normalized kind of an ``httpx`` transport exception.

    Args:
        error: The exception of the ``httpx`` client.

    Returns:
        The kind of the first class of the exception, from the most specific, found in the table.
    """
    for cls in type(error).__mro__:
        kind = _FAILURE_KINDS.get(cls)
        if kind is not None:
            return kind
    return TransportFailureKind.UNKNOWN_TRANSPORT_FAILURE


def _cap(value: float | None, remaining_s: float) -> float:
    """Cut a timeout by the remaining time; a missing timeout becomes the remaining time."""
    return remaining_s if value is None else min(value, remaining_s)


def _bounded_timeout(configured: httpx.Timeout, remaining_s: float) -> httpx.Timeout:
    """Return the client timeouts cut by the remaining time of the call, at least 1 ms each.

    Args:
        configured: The regular timeouts of the client.
        remaining_s: Seconds left until the overall deadline of the call.
    """
    remaining_s = max(0.001, remaining_s)
    return httpx.Timeout(
        connect=_cap(configured.connect, remaining_s),
        read=_cap(configured.read, remaining_s),
        write=_cap(configured.write, remaining_s),
        pool=_cap(configured.pool, remaining_s),
    )


@dataclass(frozen=True, slots=True)
class HttpxTransport:
    """The real network implementation of ``HttpTransport`` over ``httpx.Client``.

    It limits the regular client timeouts by the remaining time of the whole call, but it
    is not a mechanism for forcibly interrupting a network request already in progress:
    synchronous ``httpx`` does not guarantee that.
    """

    client: httpx.Client

    def send(self, request: HttpRequest, context: CallContext) -> HttpResponse:
        """Send the request with the client timeouts cut by the remaining time of the call.

        Raises:
            TransportFailure: ``httpx`` raised a transport exception; its kind is normalized.
        """
        started = time.perf_counter()
        try:
            response = self.client.request(
                request.method,
                request.url,
                headers=dict(request.headers),
                content=request.body,
                timeout=_bounded_timeout(self.client.timeout, context.remaining_s),
            )
        except httpx.TransportError as cause:
            kind = _failure_kind(cause)
            failure = TransportFailure(
                f"HTTP request failed: {kind.value}",
                kind=kind,
                elapsed_s=time.perf_counter() - started,
            )
            raise failure from cause
        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response.content,
            elapsed_s=time.perf_counter() - started,
        )
