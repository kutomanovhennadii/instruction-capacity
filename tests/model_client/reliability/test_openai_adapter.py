"""Tests of the OpenAI provider adapter: one attempt, error semantics.

``OpenAIProviderAdapter.generate_once`` does not retry the request - in every scenario the
transport is called exactly once. Retries are checked separately in
``test_reliable_client.py`` and in the boundary tests of ``OpenAIModelClient``.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from iec.contracts import GenParams, Prompt
from iec.model_client.openai_client import OpenAIProviderAdapter
from iec.model_client.reliability.failures import ProviderFailure, ProviderFailureKind
from iec.model_client.reliability.http_transport import HttpxTransport
from iec.model_client.reliability.reliable_client import CallContext

MODEL_ID = "gpt-4o-mini-2024-07-18"


def _context(remaining_s: float = 60.0) -> CallContext:
    return CallContext(call_id="test-call", deadline_monotonic=remaining_s, clock=lambda: 0.0)


def _success_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "lake cloud stone"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
    }


def _build_adapter(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAIProviderAdapter:
    client = httpx.Client(
        base_url="https://api.openai.com/v1",
        transport=httpx.MockTransport(handler),
    )
    return OpenAIProviderAdapter(model_id=MODEL_ID, transport=HttpxTransport(client=client))


def test_200_returns_raw_output_with_single_transport_call() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=_success_body())

    adapter = _build_adapter(handler)
    output = adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert output.text == "lake cloud stone"
    assert output.usage is not None
    assert output.usage.prompt_tokens == 17
    assert len(calls) == 1


def test_401_raises_authentication_failed() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            401, json={"error": {"code": "invalid_api_key", "message": "bad key"}}
        )

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.AUTHENTICATION_FAILED
    assert not caught.value.retry_hint.retryable
    assert len(calls) == 1


def test_404_raises_model_not_found() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            404, json={"error": {"code": "model_not_found", "message": "no such model"}}
        )

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.MODEL_NOT_FOUND
    assert not caught.value.retry_hint.retryable
    assert len(calls) == 1


def test_429_rate_limit_is_retryable_with_retry_after() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            429,
            headers={"Retry-After": "12"},
            json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
        )

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.RATE_LIMITED
    assert caught.value.retry_hint.retryable
    assert caught.value.retry_hint.retry_after_s == 12.0
    assert len(calls) == 1


def test_429_quota_is_not_retryable() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota", "message": "no credits"}},
        )

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.QUOTA_EXHAUSTED
    assert not caught.value.retry_hint.retryable
    assert len(calls) == 1


def test_503_is_provider_overloaded_and_retryable() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "service unavailable"}})

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.PROVIDER_OVERLOADED
    assert caught.value.retry_hint.retryable
    assert len(calls) == 1


def test_non_json_success_body_is_invalid_response() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(200, text="not a json body")

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.INVALID_RESPONSE
    assert len(calls) == 1


def test_success_body_without_choices_is_invalid_response() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion"})

    adapter = _build_adapter(handler)
    with pytest.raises(ProviderFailure) as caught:
        adapter.generate_once(Prompt(text="lake"), GenParams(), _context())

    assert caught.value.kind is ProviderFailureKind.INVALID_RESPONSE
    assert len(calls) == 1
