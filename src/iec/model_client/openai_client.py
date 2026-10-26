"""The OpenAI Chat Completions adapter behind the ``ModelClient`` contract.

The live ``ModelClient`` of the bench. The transport is httpx, not the provider SDK: the
nominal values of timeout, pauses and retries belong to the bench, and the raw JSON of the
response is kept in ``RawOutput.raw`` without intermediate objects of a foreign library.
Parameter names and response fields are translated here, at the boundary: foreign names do
not leak into the bench.

Network retries, the overall time limit of a call, cancellation, a simple circuit breaker
and a snapshot of active network calls are not implemented in this file: they belong to the
common layer :mod:`iec.model_client.reliability`, reusable by any future provider. This
module is exactly what knows the specifics of OpenAI: how to
assemble the request body, how to name the answer limit in it, how to parse a successful
and a failed response and which failure of this provider is cured by what.
``OpenAIProviderAdapter`` makes exactly one attempt; it is retried by
:class:`~iec.model_client.reliability.reliable_client.ReliableModelClient`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from iec.contracts import GenParams, Prompt, RawOutput, TokenUsage
from iec.infrastructure.telemetry import TelemetryContext, telemetry_step
from iec.model_client.reliability import (
    ActiveCall,
    CallContext,
    HttpRequest,
    HttpTransport,
    HttpxTransport,
    ProviderFailure,
    ProviderFailureKind,
    ReliableModelClient,
    RetryHint,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Request timeouts, seconds. The read timeout is set not by a "network" consideration but
#: by the subject: the model must write out the reference of the task in full, that is
#: generate as many tokens as there are in the text, and the timeout must not cut off a
#: model that still copes in substance. It is also a measure of the instrument's patience,
#: not a property of the model: a call silent for much longer than the worst successful one
#: is a hung call, not a slow one. Waiting is not billed: the provider charges for the
#: actual tokens.
#:
#: Reading the response: 300 s. A limit of 180 s would coincide with the ceiling of
#: gpt-4o-mini, whose answer of 16 384 tokens takes 173-179 s; gpt-5.4-mini produces 130-214
#: tokens per second with a ceiling of 128 000 and manages 39-64 thousand tokens in 300 s. A
#: run that does not fit into the limit is not a failure: it is recorded as a collapse of the
#: 3rd kind from the first word.
#:
#: Establishing the connection: 30 s - a handshake does not need more, and the split keeps
#: slow generation from masking a dead network.
DEFAULT_READ_TIMEOUT_S = 300.0
DEFAULT_CONNECT_TIMEOUT_S = 30.0
#: Compatibility for callers: the single parameter is still accepted and means the **read**
#: timeout.
DEFAULT_TIMEOUT_S = DEFAULT_READ_TIMEOUT_S
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 1.0

CHAT_COMPLETIONS_PATH = "/chat/completions"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
INSUFFICIENT_QUOTA_CODE = "insufficient_quota"

MODULE_NAME = "openai_client"

PROVIDER_ID = "openai"

_STATUS_KINDS: dict[int, ProviderFailureKind] = {
    401: ProviderFailureKind.AUTHENTICATION_FAILED,
    403: ProviderFailureKind.PERMISSION_DENIED,
    404: ProviderFailureKind.MODEL_NOT_FOUND,
    429: ProviderFailureKind.RATE_LIMITED,
    500: ProviderFailureKind.PROVIDER_OVERLOADED,
    502: ProviderFailureKind.PROVIDER_OVERLOADED,
    503: ProviderFailureKind.PROVIDER_OVERLOADED,
    504: ProviderFailureKind.PROVIDER_OVERLOADED,
}


def build_openai_http_client(
    api_key: str,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Assemble one ``httpx.Client`` per series with the key in the headers.

    The client is created once by the provider factory and lives as a field of the adapter:
    keep-alive reuses the TCP+TLS connection between points, while a client per call would
    add a handshake to every point of the series.

    Args:
        api_key: The OpenAI API key. It goes into the ``Authorization`` header of this client
            and is not stored on the adapter at all, so it cannot get into snapshots and
            logs.
        base_url: Root of the API; overridden for tests and compatible deployments.
        timeout_s: The **read** timeout of the response; for establishing the connection
            ``DEFAULT_CONNECT_TIMEOUT_S`` is taken (or ``timeout_s`` itself if smaller).
        transport: The test door - ``httpx.MockTransport`` instead of the real network.

    Returns:
        The configured client; its owner closes it (see :meth:`OpenAIModelClient.close`).

    Side effects:
        Makes no network calls: the connection is opened by the first request.
    """
    return httpx.Client(
        base_url=base_url,
        # Separate timeouts: the connection must be established fast, while reading the
        # response is legitimately long (generating ten thousand tokens takes minutes).
        timeout=httpx.Timeout(timeout_s, connect=min(DEFAULT_CONNECT_TIMEOUT_S, timeout_s)),
        transport=transport,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )


def _openai_telemetry_context(
    client: OpenAIModelClient,
    prompt: Prompt,
    params: GenParams,
) -> TelemetryContext:
    """Extract the call context of the live model available to the adapter."""
    del prompt, params
    return TelemetryContext(model_id=client.model_id)


#: OpenAI model families that reject sampling parameters. Reasoning models generate the
#: answer through internal rounds of reasoning and selection, so ``temperature``/``top_p``
#: are switched off for them: a request with them returns 400 ("Only the default (1) value
#: is supported"), that is **every** point of the series would fail. Checked by prefix, not
#: by the exact ``model_id``: a series measures a pinned revision
#: (``gpt-5.4-mini-2026-03-17``), and a list of revisions here would duplicate the registry.
_SAMPLING_FREE_PREFIXES: tuple[str, ...] = ("gpt-5", "o1", "o3", "o4")


#: Models to which the reasoning level is sent explicitly and equal to ``none``.
#: gpt-5.4-mini can reason hidden before answering; by default the API does not do it, but a
#: measurement condition must not depend on a provider default. The prefix is narrow: the
#: value ``none`` is documented only for this family; early gpt-5 models do not know it.
_REASONING_NONE_PREFIXES: tuple[str, ...] = ("gpt-5.4",)


def _rejects_sampling_params(model_id: str) -> bool:
    """Whether the model rejects ``temperature``/``top_p`` (reasoning families)."""
    return model_id.startswith(_SAMPLING_FREE_PREFIXES)


def _build_request_payload(model_id: str, prompt: Prompt, params: GenParams) -> dict[str, Any]:
    """Translate the project prompt and parameters into a Chat Completions request body.

    The prompt goes as the only user message: adding a system text would change the
    measurement conditions, and the prompt template is fixed by the protocol. The answer
    limit is named ``max_completion_tokens`` in the API (the former ``max_tokens`` is
    deprecated); the contract ``GenParams.max_tokens`` is not renamed - translating the name
    is exactly the duty of the boundary.

    Sampling parameters are not sent to every model: reasoning families reject them (see
    :data:`_SAMPLING_FREE_PREFIXES`), and sending them would bring down every point of the
    series with error 400. The decision is made by the adapter, not by the series config:
    the ``GenParams`` contract explicitly assigns "handling unsupported parameters" to the
    model adapter, and the nominal values of a series stay the same for all models -
    otherwise the configs of pilots would drift apart by provider properties.

    This does not change the measurement conditions: such a model has no tunable sampling
    anyway, and ``temperature = 0`` in the config meant exactly "no sampling".

    Models of the gpt-5.4 family get the reasoning level ``none`` explicitly
    (:data:`_REASONING_NONE_PREFIXES`): capacity is measured without hidden reasoning, as for
    models that do not have it.
    """
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt.text}],
        "max_completion_tokens": params.max_tokens,
    }
    if not _rejects_sampling_params(model_id):
        payload["temperature"] = params.temperature
        payload["top_p"] = params.top_p
    if model_id.startswith(_REASONING_NONE_PREFIXES):
        payload["reasoning_effort"] = "none"
    # The OpenAI seed is best-effort determinism; an unset seed is not sent at all, so as
    # not to impose on the model a value the protocol did not require.
    if params.seed is not None:
        payload["seed"] = params.seed
    return payload


def _safe_json_body(body: bytes) -> Any:
    """Parse the response body as JSON, returning ``None`` instead of an exception."""
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value


def _text_head(body: bytes) -> str:
    """The first 200 characters of the body as text - for an error message."""
    return body.decode("utf-8", errors="replace")[:200]


def _parse_usage(usage: Any) -> TokenUsage | None:
    """Translate the provider token counter into the project fields.

    ``None`` means "the provider did not report the counter", not zero tokens, so missing
    quantities stay empty.
    """
    if not isinstance(usage, dict):
        return None
    prompt_tokens = usage.get("prompt_tokens")
    response_tokens = usage.get("completion_tokens")
    return TokenUsage(
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        response_tokens=response_tokens if isinstance(response_tokens, int) else None,
    )


def _parse_response_body(body_bytes: bytes, latency_s: float) -> RawOutput:
    """Translate a successful Chat Completions response into ``RawOutput``.

    Absence of text in the response (``content: null`` when the model refuses) is data, not
    a failure: the validator scores such an answer with zeros. A violation of the response
    form itself is a failure.

    Raises:
        ProviderFailure: ``INVALID_RESPONSE`` - the body is not JSON or has no ``choices``,
            that is the response does not follow the API form.
    """
    body = _safe_json_body(body_bytes)
    if not isinstance(body, dict):
        raise ProviderFailure(
            "OpenAI response body is not a JSON object",
            provider=PROVIDER_ID,
            kind=ProviderFailureKind.INVALID_RESPONSE,
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderFailure(
            "OpenAI response has no usable 'choices' entry",
            provider=PROVIDER_ID,
            kind=ProviderFailureKind.INVALID_RESPONSE,
        )
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = choice.get("finish_reason")
    return RawOutput(
        text=content if isinstance(content, str) else "",
        usage=_parse_usage(body.get("usage")),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        latency_s=latency_s,
        # The whole snapshot: system_fingerprint, refusal and the request identifier are
        # needed by the API log and stay available for audit without separate contract
        # fields.
        raw=body,
    )


def _parse_retry_after(header_value: str | None) -> float | None:
    """Read ``Retry-After`` in seconds, ignoring the non-numeric form of the header."""
    if header_value is None:
        return None
    try:
        return float(header_value.strip())
    except ValueError:
        # The date form (HTTP-date) is not supported by the bench: its own nominal pause is
        # certainly no worse, and parsing a date would add a point of failure at the
        # boundary.
        return None


def _describe_failure(model_id: str, response: Any) -> ProviderFailure:
    """Classify an unsuccessful OpenAI HTTP response into ``ProviderFailure``.

    Lack of credits comes with the same status 429 as a rate limit, but a retry does not
    cure it: the error code in the body is the only way to tell them apart. The adapter
    must know this difference and nobody above it.
    """
    body = _safe_json_body(response.body)
    error = body.get("error") if isinstance(body, dict) else None
    error_code = error.get("code") if isinstance(error, dict) else None
    error_message = error.get("message") if isinstance(error, dict) else None
    detail = str(error_message) if error_message else _text_head(response.body)
    message = (
        f"OpenAI request for model '{model_id}' failed with HTTP {response.status_code}: {detail}"
    )

    if error_code == INSUFFICIENT_QUOTA_CODE:
        return ProviderFailure(
            message,
            provider=PROVIDER_ID,
            kind=ProviderFailureKind.QUOTA_EXHAUSTED,
            http_status=response.status_code,
            retry_hint=RetryHint(retryable=False),
            provider_code=error_code,
        )

    kind = _STATUS_KINDS.get(response.status_code, ProviderFailureKind.INVALID_REQUEST)
    retryable = response.status_code in RETRYABLE_STATUS_CODES
    # HttpResponse.headers comes from httpx.Response.headers through dict(...): httpx
    # normalizes header names to lower case in such materialization, so the key is looked
    # up in lower case regardless of the case in which the server sent it.
    retry_after_s = _parse_retry_after(response.headers.get("retry-after")) if retryable else None
    return ProviderFailure(
        message,
        provider=PROVIDER_ID,
        kind=kind,
        http_status=response.status_code,
        retry_hint=RetryHint(retryable=retryable, retry_after_s=retry_after_s),
        provider_code=error_code if isinstance(error_code, str) else None,
    )


@dataclass(frozen=True, slots=True)
class OpenAIProviderAdapter:
    """The OpenAI provider adapter: knows the request format, the response and error semantics.

    It makes exactly one attempt per :meth:`generate_once` call - retries, the overall time
    limit and the circuit breaker belong to :class:`ReliableModelClient`, which holds this
    adapter as a field.
    """

    model_id: str
    transport: HttpTransport

    @property
    def provider_id(self) -> str:
        return PROVIDER_ID

    def generate_once(self, prompt: Prompt, params: GenParams, context: CallContext) -> RawOutput:
        payload = _build_request_payload(self.model_id, prompt, params)
        response = self.transport.send(
            HttpRequest(
                method="POST",
                url=CHAT_COMPLETIONS_PATH,
                body=json.dumps(payload).encode("utf-8"),
            ),
            context,
        )
        if response.status_code >= 400:
            raise _describe_failure(self.model_id, response)
        return _parse_response_body(response.body, response.elapsed_s)


@dataclass(slots=True)
class OpenAIModelClient:
    """The Chat Completions adapter behind the ``ModelClient`` contract.

    Owns one long-lived :class:`ReliableModelClient`, created in :meth:`__post_init__` and
    living for the whole series: only this way the circuit breaker and the observability of
    active calls stay common for all points of the series instead of being reset on every
    call. Both test doors are set by the constructor - the transport when assembling the
    ``httpx.Client`` and the retry pause by the :attr:`sleep` field.
    """

    model_id: str
    client: httpx.Client
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_s: float = DEFAULT_BACKOFF_BASE_S
    sleep: Callable[[float], None] = field(default=time.sleep)
    _reliable: ReliableModelClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._reliable = ReliableModelClient(
            adapter=OpenAIProviderAdapter(
                model_id=self.model_id,
                transport=HttpxTransport(client=self.client),
            ),
            max_retries=self.max_retries,
            backoff_base_s=self.backoff_base_s,
            sleep=self.sleep,
        )

    @telemetry_step(
        "model_generate",
        module_name=MODULE_NAME,
        context_extractor=_openai_telemetry_context,
    )
    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        """Make one Chat Completions call and return the raw answer as a project type.

        Args:
            prompt: The ready prompt of the task; goes as one user message without a system
                role, because the prompt template is a protocol constant.
            params: Technical parameters of the call; translated into API names at the
                boundary.

        Returns:
            ``RawOutput`` with the answer text, the token counter in project fields, the
            finish reason as is, the latency of the last attempt and the parsed response
            JSON.

        Raises:
            ProviderError: ``PROVIDER_TIMEOUT`` - the request timeout or the overall time
                limit of the call expired; ``PROVIDER_API_ERROR`` - retries on
                429/5xx/network failure are exhausted, or an error that a retry does not
                cure (lack of credits, other 4xx); ``PROVIDER_RESPONSE_INVALID`` - a
                successful response cannot be parsed.

        Side effects:
            Calls the OpenAI API, spends the organization's credits and pauses between
            retries.
        """
        return self._reliable.generate(prompt, params)

    def cancel(self, reason: str) -> None:
        """Cancel current and future retries of this client (``ReliableModelClient.cancel``)."""
        self._reliable.cancel(reason)

    def in_flight_snapshot(self) -> tuple[ActiveCall, ...]:
        """Return a snapshot of the network calls of this client that have not finished yet."""
        return self._reliable.in_flight_snapshot()

    def close(self) -> None:
        """Close the connections of the httpx client.

        The method is not in the ``ModelClient`` contract, and the runner does not call it:
        in a series the sockets are closed by the finishing process. The owners of the call
        are the tests and a live smoke run, where a client not collected by the garbage
        collector would give a ``ResourceWarning``.

        Side effects:
            Closes the connection pool; a repeated call is safe.
        """
        self.client.close()
