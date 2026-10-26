"""Mirror tests of the live OpenAI Chat Completions adapter.

All tests except the smoke tests marked ``network`` run without network: the transport is
replaced by ``httpx.MockTransport`` when the client is built, retry pauses by a stub of the
``sleep`` field. The fixture builds the client and closes it in teardown: an
``httpx.Client`` not collected by the garbage collector gives a ``ResourceWarning`` in a
pytest run.
"""

import json
import os
from collections.abc import Callable, Iterator

import httpx
import pytest

from iec.contracts import GenParams, Prompt, RawOutput, TokenUsage
from iec.infrastructure.exceptions import ProviderError
from iec.model_client import ModelClient
from iec.model_client.openai_client import (
    DEFAULT_MAX_RETRIES,
    OpenAIModelClient,
    build_openai_http_client,
)

PILOT_MODEL_ID = "gpt-4o-mini-2024-07-18"
SMOKE_MODEL_ID = "gpt-4o-mini-2024-07-18"


class SleepSpy:
    """A pause stub: collects the requested intervals instead of waiting."""

    def __init__(self) -> None:
        self.pauses: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.pauses.append(seconds)


def _success_body(
    content: str | None = "lake cloud stone",
    finish_reason: str = "stop",
) -> dict[str, object]:
    """Assemble a successful Chat Completions response body in the form of the real API."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": PILOT_MODEL_ID,
        "system_fingerprint": "fp_test_1234",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
    }


@pytest.fixture
def adapter_factory() -> Iterator[Callable[..., OpenAIModelClient]]:
    """Build an adapter on the given request handler and close the client in teardown."""
    built: list[OpenAIModelClient] = []

    def build(
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        sleep: Callable[[float], None] | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        model_id: str = PILOT_MODEL_ID,
    ) -> OpenAIModelClient:
        http_client = build_openai_http_client(
            "sk-test-key",
            transport=httpx.MockTransport(handler),
        )
        adapter = OpenAIModelClient(
            model_id=model_id,
            client=http_client,
            max_retries=max_retries,
            sleep=sleep if sleep is not None else SleepSpy(),
        )
        built.append(adapter)
        return adapter

    try:
        yield build
    finally:
        for adapter in built:
            adapter.close()


def test_successful_call_translates_request_and_response(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body())

    adapter = adapter_factory(handler)
    prompt = Prompt(text="Replace every 'lake' with 'cloud'.")

    output = adapter.generate(prompt, GenParams(temperature=0.0, max_tokens=256, top_p=1.0))

    # The request: the only user message with the exact prompt text, without a system role.
    assert len(seen) == 1
    assert seen[0].url.path.endswith("/chat/completions")
    payload = json.loads(seen[0].content)
    assert payload["messages"] == [{"role": "user", "content": prompt.text}]
    assert payload["model"] == PILOT_MODEL_ID
    # The name of the answer limit is translated at the boundary; GenParams is not renamed.
    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    # An unset seed is not sent at all.
    assert "seed" not in payload
    # The key goes as a client header and is not stored on the adapter.
    assert seen[0].headers["Authorization"] == "Bearer sk-test-key"

    assert isinstance(output, RawOutput)
    assert output.text == "lake cloud stone"
    assert output.usage == TokenUsage(prompt_tokens=17, response_tokens=3)
    assert output.finish_reason == "stop"
    assert output.latency_s is not None and output.latency_s >= 0.0
    assert output.raw["system_fingerprint"] == "fp_test_1234"


def test_seed_is_sent_when_set(adapter_factory: Callable[..., OpenAIModelClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body())

    adapter_factory(handler).generate(Prompt(text="lake"), GenParams(seed=42))

    assert json.loads(seen[0].content)["seed"] == 42


def test_null_content_becomes_empty_text_without_raising(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_success_body(content=None))

    output = adapter_factory(handler).generate(Prompt(text="lake"), GenParams())

    # A model refusal is data: the validator scores an empty answer with zeros, not an error.
    assert output.text == ""
    assert output.usage == TokenUsage(prompt_tokens=17, response_tokens=3)


def test_length_finish_reason_passes_through(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_success_body(finish_reason="length"))

    output = adapter_factory(handler).generate(Prompt(text="lake"), GenParams())

    # The finish reason goes as is.
    assert output.finish_reason == "length"


def test_timeout_stops_the_call_without_a_retry(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("timed out", request=request)

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy)

    with pytest.raises(ProviderError) as caught:
        adapter.generate(Prompt(text="lake"), GenParams())

    assert caught.value.error_code == "PROVIDER_TIMEOUT"
    # There is no retry on timeout (``TIMEOUT_RETRIES = 0``): the model did not fit into the
    # allotted time on this node, and a second attempt would give the same answer in the
    # same minutes: a retry would double the cost of a heavy point.
    assert len(calls) == 1
    assert sleep_spy.pauses == []


def test_rate_limit_retries_and_respects_retry_after(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "7"},
                json={"error": {"code": "rate_limit_exceeded", "message": "slow down"}},
            )
        return httpx.Response(200, json=_success_body())

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy)

    output = adapter.generate(Prompt(text="lake"), GenParams())

    assert output.text == "lake cloud stone"
    assert len(calls) == 2
    # The pause is not shorter than requested by the server: our nominal first pause is 1 s.
    assert sleep_spy.pauses == [7.0]


def test_insufficient_quota_fails_without_retries(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            429,
            json={"error": {"code": "insufficient_quota", "message": "no credits left"}},
        )

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy)

    with pytest.raises(ProviderError) as caught:
        adapter.generate(Prompt(text="lake"), GenParams())

    # Credits are over - a retry does not cure it, there is nothing to retry.
    assert caught.value.error_code == "PROVIDER_API_ERROR"
    assert len(calls) == 1
    assert sleep_spy.pauses == []


def test_server_errors_exhaust_retries(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "service unavailable"}})

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy, max_retries=3)

    with pytest.raises(ProviderError) as caught:
        adapter.generate(Prompt(text="lake"), GenParams())

    assert caught.value.error_code == "PROVIDER_API_ERROR"
    assert len(calls) == 1 + 3
    # Exponential pauses between attempts; the last attempt does not end with a pause.
    assert sleep_spy.pauses == [1.0, 2.0, 4.0]


def test_transport_error_is_retried(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json=_success_body())

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy)

    output = adapter.generate(Prompt(text="lake"), GenParams())

    assert output.text == "lake cloud stone"
    assert len(calls) == 2
    assert sleep_spy.pauses == [1.0]


def test_unauthorized_fails_without_retries(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls.append(1)
        return httpx.Response(
            401,
            json={"error": {"code": "invalid_api_key", "message": "Incorrect API key"}},
        )

    sleep_spy = SleepSpy()
    adapter = adapter_factory(handler, sleep=sleep_spy)

    with pytest.raises(ProviderError) as caught:
        adapter.generate(Prompt(text="lake"), GenParams())

    # The key or the request - a retry does not cure it.
    assert caught.value.error_code == "PROVIDER_API_ERROR"
    assert len(calls) == 1
    assert sleep_spy.pauses == []


def test_malformed_success_body_is_response_invalid(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"id": "chatcmpl-test", "object": "chat.completion"})

    with pytest.raises(ProviderError) as caught:
        adapter_factory(handler).generate(Prompt(text="lake"), GenParams())

    # The API response form is broken - a boundary failure, not bad model data.
    assert caught.value.error_code == "PROVIDER_RESPONSE_INVALID"


def test_non_json_success_body_is_response_invalid(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, text="not a json body")

    with pytest.raises(ProviderError) as caught:
        adapter_factory(handler).generate(Prompt(text="lake"), GenParams())

    assert caught.value.error_code == "PROVIDER_RESPONSE_INVALID"


def test_adapter_satisfies_model_client_protocol(
    adapter_factory: Callable[..., OpenAIModelClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_success_body())

    assert isinstance(adapter_factory(handler), ModelClient)


@pytest.mark.network
def test_live_smoke_call_returns_usable_answer() -> None:
    """A live call of the pilot model: one tiny prompt, a check of the answer form.

    It also checks that the 2024 snapshot accepts ``max_completion_tokens``: a rejected
    parameter name would give HTTP 400 and ``PROVIDER_API_ERROR``.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set - live smoke test skipped")

    http_client = build_openai_http_client(api_key)
    adapter = OpenAIModelClient(model_id=SMOKE_MODEL_ID, client=http_client)
    try:
        output = adapter.generate(
            Prompt(text="Reply with the single word: lake"),
            GenParams(temperature=0.0, max_tokens=16),
        )
    except ProviderError as error:
        pytest.skip(f"OpenAI API is unavailable: {error.error_code} {error}")
    finally:
        adapter.close()

    assert output.text.strip() != ""
    assert output.usage is not None
    assert output.usage.prompt_tokens is not None and output.usage.prompt_tokens > 0
    assert output.usage.response_tokens is not None and output.usage.response_tokens > 0
    assert output.finish_reason == "stop"


def test_reasoning_models_get_no_sampling_params(adapter_factory) -> None:
    """A reasoning model gets no temperature/top_p: it rejects them with error 400.

    A series sends the same nominal values to all models (measurement conditions belong to
    the series, not to the provider), so selecting parameters is the adapter's duty: the
    ``GenParams`` contract explicitly assigns "handling unsupported parameters" to it.
    Without this selection **every** point of a series on gpt-5.4-mini would fail, not one.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body())

    adapter = adapter_factory(handler, model_id="gpt-5.4-mini-2026-03-17")
    adapter.generate(Prompt(text="x"), GenParams(temperature=0.0, max_tokens=64, top_p=1.0))

    payload = json.loads(seen[0].content)
    assert "temperature" not in payload
    assert "top_p" not in payload
    # The rest of the body is unchanged: the selection concerns sampling only.
    assert payload["model"] == "gpt-5.4-mini-2026-03-17"
    assert payload["max_completion_tokens"] == 64
    assert payload["messages"] == [{"role": "user", "content": "x"}]


def test_gpt_5_4_gets_reasoning_effort_none(adapter_factory) -> None:
    """gpt-5.4 gets ``reasoning_effort = none``, gpt-4o-mini gets none at all."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_success_body())

    for model_id in ("gpt-5.4-mini-2026-03-17", "gpt-4o-mini-2024-07-18"):
        adapter = adapter_factory(handler, model_id=model_id)
        adapter.generate(Prompt(text="x"), GenParams(temperature=0.0, max_tokens=64))

    reasoning, plain = (json.loads(request.content) for request in seen)
    assert reasoning["reasoning_effort"] == "none"
    assert "reasoning_effort" not in plain


@pytest.mark.network
def test_live_gpt_5_4_mini_accepts_reasoning_effort_none() -> None:
    """A live call of gpt-5.4-mini with ``reasoning_effort = none``: the API accepts it.

    A provider failure is not turned into a skip here: it is exactly what the test checks.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not set - live test skipped")
    adapter = OpenAIModelClient(
        model_id="gpt-5.4-mini-2026-03-17", client=build_openai_http_client(api_key)
    )
    try:
        output = adapter.generate(
            Prompt(text="Reply with the single word: lake"),
            GenParams(temperature=0.0, max_tokens=16),
        )
    finally:
        adapter.close()
    assert output.text.strip() != ""
