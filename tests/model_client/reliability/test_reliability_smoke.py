"""An end-to-end smoke test of the reliability layer: the whole wiring, without network.

Only the wiring ``build_openai_provider -> OpenAIModelClient -> ReliableModelClient ->
OpenAIProviderAdapter -> HttpxTransport -> MockTransport -> RawOutput`` is checked. The
scenario is one HTTP 200 with a valid Chat Completions body. Retries, 503 and the circuit
breaker are not checked here: there are separate unit tests for them in
``test_reliable_client.py`` and ``test_openai_adapter.py``. The test takes a fraction of a
second.
"""

from __future__ import annotations

import httpx
import pytest

from iec.contracts import GenParams, Prompt, RawOutput
from iec.model_client import model_builder
from iec.model_client.model_builder import build_openai_provider, OPENAI_API_KEY_ENV
from iec.model_client.openai_client import OpenAIModelClient
from iec.model_client.reliability import HttpxTransport
from iec.model_client.reliability.reliable_client import ReliableModelClient
from iec.model_client.tiktoken_tokenizer import TiktokenTokenizer

PILOT_MODEL_ID = "gpt-4o-mini-2024-07-18"


def _success_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-smoke",
        "object": "chat.completion",
        "model": PILOT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "lake cloud stone"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
    }


def test_build_openai_provider_wires_reliability_stack_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One artificial HTTP 200 goes through the whole wiring and gives a correct ``RawOutput``.

    The transport is replaced at the level of the ``httpx.Client`` that
    ``build_openai_http_client`` builds inside ``build_openai_provider`` - the provider factory
    itself and the whole path through the layers run as in production, only the network end
    point changes.
    """
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "sk-test-key")

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_success_body())

    original_build_http_client = model_builder.build_openai_http_client

    def build_http_client_with_mock_transport(api_key: str) -> httpx.Client:
        return original_build_http_client(api_key, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(
        model_builder, "build_openai_http_client", build_http_client_with_mock_transport
    )

    from iec.contracts import ModelConfig

    config = ModelConfig(
        model_id=PILOT_MODEL_ID,
        provider="openai",
        gen_params=GenParams(),
        tokenizer_name="gpt-4o-mini",
    )

    client, tokenizer = build_openai_provider(config)
    try:
        # The whole wiring: OpenAIModelClient owns ReliableModelClient, whose adapter uses
        # HttpxTransport with the replaced transport.
        assert isinstance(client, OpenAIModelClient)
        assert isinstance(tokenizer, TiktokenTokenizer)
        assert isinstance(client._reliable, ReliableModelClient)
        assert isinstance(client._reliable.adapter.transport, HttpxTransport)  # type: ignore[attr-defined]

        output = client.generate(Prompt(text="lake"), GenParams(temperature=0.0, max_tokens=16))

        assert isinstance(output, RawOutput)
        assert output.text == "lake cloud stone"
        assert output.usage is not None
        assert output.usage.prompt_tokens == 17
        assert output.usage.response_tokens == 3
        assert output.finish_reason == "stop"
        assert len(calls) == 1
    finally:
        client.close()
