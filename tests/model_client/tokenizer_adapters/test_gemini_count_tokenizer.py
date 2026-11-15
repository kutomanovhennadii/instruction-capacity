"""The tokenizer of a Gemini model by its own count; the provider is a test double."""

import json

import httpx
import pytest

from iec.infrastructure.exceptions import ProviderError
from iec.model_client.tokenizer_adapters.gemini_count_tokenizer import (
    GeminiCountTokenizer,
    build_gemini_count_client,
    build_gemini_count_tokenizer,
)


def _word_counter(request: httpx.Request) -> httpx.Response:
    """A provider double: as many tokens as words in the text."""
    text = json.loads(request.content)["contents"][0]["parts"][0]["text"]
    return httpx.Response(200, json={"totalTokens": len(text.split())})


def _count_client(handler) -> httpx.Client:  # noqa: ANN001 - a handler of MockTransport
    return build_gemini_count_client(
        "test-key", base_url="https://gemini.test/v1beta", transport=httpx.MockTransport(handler)
    )


def test_single_tokenness_is_the_difference_of_two_counts_and_probes_are_cached() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _word_counter(request)

    tokenizer = build_gemini_count_tokenizer("gemini-3.5-flash-lite", _count_client(handler))
    assert tokenizer.name == "gemini_count/gemini-3.5-flash-lite"
    assert tokenizer.is_single_token(" lake") is True
    assert tokenizer.is_single_token(" lake stone") is False
    # The anchor is counted once: two probes and one anchor are three requests.
    assert len(requests) == 3
    assert requests[0].url.path == "/v1beta/models/gemini-3.5-flash-lite:countTokens"
    assert requests[0].headers["x-goog-api-key"] == "test-key"


def test_a_long_text_is_counted_anew_and_a_refusal_is_retried() -> None:
    calls: list[int] = []
    pauses: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={})
        return _word_counter(request)

    tokenizer = GeminiCountTokenizer(
        name="gemini_count/m", model_id="m", client=_count_client(handler), sleep=pauses.append
    )
    text = "lake " * 40
    assert tokenizer.count_tokens(text) == 40
    assert tokenizer.count_tokens(text) == 40
    assert len(calls) == 3
    assert pauses == [2.0]


def test_a_request_error_of_count_tokens_is_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad"}})

    tokenizer = build_gemini_count_tokenizer("m", _count_client(handler))
    with pytest.raises(ProviderError) as failure:
        tokenizer.count_tokens("lake")
    assert failure.value.error_code == "GEMINI_COUNT_FAILED"
