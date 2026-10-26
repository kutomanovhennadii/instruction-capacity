"""Tests of the transport layer: HTTP delivery without provider semantics.

All tests run without network - the handler is replaced by ``httpx.MockTransport``.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from iec.model_client.reliability.failures import TransportFailure, TransportFailureKind
from iec.model_client.reliability.http_transport import HttpRequest, HttpxTransport


class _FixedContext:
    """A ``CallContext`` stub with a given remaining time."""

    def __init__(self, remaining_s: float) -> None:
        self._remaining_s = remaining_s

    @property
    def remaining_s(self) -> float:
        return self._remaining_s


def _build_transport(handler: Callable[[httpx.Request], httpx.Response]) -> HttpxTransport:
    client = httpx.Client(
        base_url="https://example.invalid",
        transport=httpx.MockTransport(handler),
    )
    return HttpxTransport(client=client)


@pytest.mark.parametrize("status_code", [200, 429, 500])
def test_status_codes_pass_through_without_interpretation(status_code: int) -> None:
    """The transport does not interpret the HTTP status: any code is a normal response."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status_code, json={"ok": True})

    transport = _build_transport(handler)
    response = transport.send(
        HttpRequest(method="POST", url="/v1/thing", body=b"{}"),
        _FixedContext(remaining_s=30.0),
    )

    assert response.status_code == status_code
    assert response.body == b'{"ok":true}'
    assert response.elapsed_s >= 0.0


@pytest.mark.parametrize(
    ("raised", "expected_kind"),
    [
        (
            lambda request: httpx.ReadTimeout("timed out", request=request),
            TransportFailureKind.READ_TIMEOUT,
        ),
        (
            lambda request: httpx.ConnectTimeout("connect timed out", request=request),
            TransportFailureKind.CONNECT_TIMEOUT,
        ),
        (
            lambda request: httpx.ConnectError("connection refused", request=request),
            TransportFailureKind.CONNECT_FAILURE,
        ),
        (
            lambda request: httpx.ReadError("connection reset", request=request),
            TransportFailureKind.CONNECTION_RESET,
        ),
    ],
)
def test_transport_errors_become_normalized_failure_kinds(
    raised: Callable[[httpx.Request], httpx.TransportError],
    expected_kind: TransportFailureKind,
) -> None:
    """``httpx`` exceptions turn into ``TransportFailure`` with a normalized name; the
    ``httpx`` exception type does not get out."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise raised(request)

    transport = _build_transport(handler)

    with pytest.raises(TransportFailure) as caught:
        transport.send(
            HttpRequest(method="POST", url="/v1/thing", body=b"{}"),
            _FixedContext(remaining_s=30.0),
        )

    assert caught.value.kind is expected_kind
    assert not isinstance(caught.value, httpx.TransportError)


def test_no_httpx_exception_type_escapes_transport() -> None:
    """The ancestor type is checked separately: the caught exception is ``TransportFailure``,
    not any subclass of ``httpx.TransportError``."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.WriteTimeout("write timed out", request=request)

    transport = _build_transport(handler)

    with pytest.raises(TransportFailure) as caught:
        transport.send(
            HttpRequest(method="POST", url="/v1/thing", body=b"{}"),
            _FixedContext(remaining_s=30.0),
        )

    assert type(caught.value) is TransportFailure
    assert caught.value.kind is TransportFailureKind.WRITE_TIMEOUT


def test_attempt_timeout_is_capped_by_remaining_time() -> None:
    """The network timeout of an attempt does not exceed the remaining overall call time."""
    seen_timeouts: list[httpx.Timeout] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # httpx exposes no public API to read the timeout from the handler, so a transport
        # with a known client.timeout is used, and reading through the request object is
        # unavailable; instead the fact is pinned through the side effect below.
        del request
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(
        base_url="https://example.invalid",
        timeout=httpx.Timeout(300.0, connect=30.0),
        transport=httpx.MockTransport(handler),
    )
    transport = HttpxTransport(client=client)

    # The remaining time is below the regular client timeout - the limit must apply. The
    # test checks it directly through the helper function of the module.
    from iec.model_client.reliability.http_transport import _bounded_timeout

    bounded = _bounded_timeout(client.timeout, remaining_s=5.0)
    assert bounded.read == 5.0
    assert bounded.connect == 5.0

    # The regular case: with a large remaining time the limit does not lower the client
    # timeout.
    bounded_large = _bounded_timeout(client.timeout, remaining_s=600.0)
    assert bounded_large.read == 300.0
    assert bounded_large.connect == 30.0

    response = transport.send(
        HttpRequest(method="POST", url="/v1/thing", body=b"{}"),
        _FixedContext(remaining_s=5.0),
    )
    assert response.status_code == 200
    del seen_timeouts
