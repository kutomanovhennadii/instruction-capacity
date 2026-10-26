"""The transport layer: HTTP delivery only, without the semantics of any provider.

``HttpTransport`` can do one thing - send a request and return the response. A response
with code 429 or 500 is a normally received response for it, not an error: interpreting
HTTP statuses is the business of the provider adapter, not of the transport. The only
error here is non-delivery, and every such outcome is named from the closed list
``TransportFailureKind`` instead of propagating as an ``httpx`` exception.
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
    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    elapsed_s: float


class CallContext(Protocol):
    @property
    def remaining_s(self) -> float: ...


class HttpTransport(Protocol):
    def send(self, request: HttpRequest, context: CallContext) -> HttpResponse: ...


def _kind(error: httpx.TransportError) -> TransportFailureKind:
    if isinstance(error, httpx.ConnectTimeout):
        return TransportFailureKind.CONNECT_TIMEOUT
    if isinstance(error, httpx.WriteTimeout):
        return TransportFailureKind.WRITE_TIMEOUT
    if isinstance(error, httpx.ReadTimeout | httpx.PoolTimeout):
        return TransportFailureKind.READ_TIMEOUT
    if isinstance(error, httpx.ConnectError):
        return TransportFailureKind.CONNECT_FAILURE
    if isinstance(error, httpx.ReadError | httpx.WriteError | httpx.CloseError):
        return TransportFailureKind.CONNECTION_RESET
    if isinstance(error, httpx.ProtocolError):
        return TransportFailureKind.REMOTE_PROTOCOL_FAILURE
    return TransportFailureKind.UNKNOWN_TRANSPORT_FAILURE


def _cap(value: float | None, remaining_s: float) -> float:
    return remaining_s if value is None else min(value, remaining_s)


def _bounded_timeout(configured: httpx.Timeout, remaining_s: float) -> httpx.Timeout:
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
            failure = TransportFailure(
                f"HTTP request failed: {_kind(cause).value}",
                kind=_kind(cause),
                elapsed_s=time.perf_counter() - started,
            )
            raise failure from cause
        return HttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=response.content,
            elapsed_s=time.perf_counter() - started,
        )
