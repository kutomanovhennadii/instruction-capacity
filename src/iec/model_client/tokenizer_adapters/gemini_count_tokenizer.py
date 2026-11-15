"""The tokenizer of a Gemini model by its own count: the method ``countTokens`` of the Gemini API.

It serves a model whose tokenizer is not published (gemini-3.5-flash-lite). A single token is
checked indirectly, by the difference of two counts: the anchor ``word`` with the probe and without
it. The word stands after another word, as in the prompt, so a normalization of the beginning of a
text cannot cut its space. The count is free; refusals 429 and 5xx are retried with a pause. Short
strings (the probes of a vocabulary) are cached, prompts are not.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from iec.infrastructure.exceptions import ProviderError

GEMINI_COUNT_PROVIDER = "gemini_count"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_KEY_HEADER = "x-goog-api-key"
READ_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 30.0
#: The word before a probe: a single token for any model, and it does not merge with the probe.
PROBE_ANCHOR = "word"
CACHED_TEXT_MAX_CHARS = 64
MAX_ATTEMPTS = 6
BACKOFF_BASE_S = 2.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(slots=True)
class GeminiCountTokenizer:
    """A tokenizer behind the ``Tokenizer`` contract that lets the model count the tokens."""

    name: str
    model_id: str
    client: httpx.Client
    sleep: Callable[[float], None] = field(default=time.sleep)
    _cache: dict[str, int] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def is_single_token(self, word: str) -> bool:
        """Return whether the string costs exactly one token: the anchor with it minus without it."""
        return self.count_tokens(PROBE_ANCHOR + word) - self.count_tokens(PROBE_ANCHOR) == 1

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens of the text by the count of the model.

        Raises:
            ProviderError: ``GEMINI_COUNT_FAILED`` - a refusal that retries did not cure;
                ``GEMINI_COUNT_INVALID`` - the answer has no ``totalTokens``.
        """
        cacheable = len(text) <= CACHED_TEXT_MAX_CHARS
        if cacheable:
            with self._lock:
                cached = self._cache.get(text)
            if cached is not None:
                return cached
        count = self._request(text)
        if cacheable:
            with self._lock:
                self._cache[text] = count
        return count

    def _request(self, text: str) -> int:
        """Ask the model to count the tokens of the text, retrying refusals a pause may cure.

        Raises:
            ProviderError: ``GEMINI_COUNT_FAILED`` - a refusal that is not retried or that the
                retries did not cure; ``GEMINI_COUNT_INVALID`` - the answer has no
                ``totalTokens``.
        """
        body = json.dumps({"contents": [{"role": "user", "parts": [{"text": text}]}]})
        last = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self.client.post(
                    f"/models/{self.model_id}:countTokens", content=body.encode("utf-8")
                )
            except httpx.TransportError as error:
                last = f"{type(error).__name__}: {error}"
            else:
                if response.status_code == 200:
                    total = _total_tokens(response)
                    if total is None:
                        raise ProviderError(
                            f"countTokens of '{self.model_id}' returned no totalTokens",
                            error_code="GEMINI_COUNT_INVALID",
                        )
                    return total
                if response.status_code not in _RETRYABLE_STATUS:
                    raise ProviderError(
                        f"countTokens of '{self.model_id}' failed with HTTP "
                        f"{response.status_code}: {response.text[:200]}",
                        error_code="GEMINI_COUNT_FAILED",
                    )
                last = f"HTTP {response.status_code}"
            if attempt < MAX_ATTEMPTS:
                self.sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
        raise ProviderError(
            f"countTokens of '{self.model_id}' failed after {MAX_ATTEMPTS} attempts: {last}",
            error_code="GEMINI_COUNT_FAILED",
        )


def _total_tokens(response: httpx.Response) -> int | None:
    """Return ``totalTokens`` of a count answer, or ``None`` when the body is not such an answer."""
    try:
        payload = response.json()
    except ValueError:
        return None
    value = payload.get("totalTokens") if isinstance(payload, dict) else None
    return value if isinstance(value, int) else None


def build_gemini_count_client(
    api_key: str,
    base_url: str = GEMINI_BASE_URL,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build the HTTP client of the count with the key in a header.

    Side effects:
        No network calls: the first request opens the connection.
    """
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        transport=transport,
        headers={API_KEY_HEADER: api_key, "Content-Type": "application/json"},
    )


def build_gemini_count_tokenizer(model_id: str, client: httpx.Client) -> GeminiCountTokenizer:
    """Build the tokenizer of a Gemini model with the name ``gemini_count/<model>``."""
    return GeminiCountTokenizer(
        name=f"{GEMINI_COUNT_PROVIDER}/{model_id}", model_id=model_id, client=client
    )
