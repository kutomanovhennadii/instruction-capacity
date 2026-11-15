# reliability — the reliable call layer of a model provider

Every call of a live model goes through this layer. It separates three concerns that a provider adapter would otherwise mix:
delivering an HTTP request, knowing the API of one provider, and the policy of retries, time limits and cancellation. A reader
needs it to know why a point ends with the status `api_error` and how long a call may take.

## What is here

| Module | Role |
|---|---|
| `failures.py` | the normalized failures and the policy of every kind: the category of the final failure, whether a retry may cure it, whether it counts against the circuit |
| `http_transport.py` | the transport layer: HTTP delivery only; an `httpx` exception becomes a `TransportFailure` of the kind found by its class |
| `reliable_client.py` | the reliability layer: a common shell over any provider adapter |

## Key entry points

```python
ReliableModelClient(adapter: ProviderAdapter, max_retries=3, backoff_base_s=1.0,
                    overall_deadline_s=300.0, failure_threshold=5, cooldown_s=30.0)
ReliableModelClient.generate(prompt: Prompt, params: GenParams) -> RawOutput
ReliableModelClient.cancel(reason: str) -> None
ReliableModelClient.in_flight_snapshot() -> tuple[ActiveCall, ...]
ProviderAdapter                    # protocol: provider_id, model_id, generate_once(prompt, params, context)
ClassifiedFailure                  # a failure with its policy: TransportFailure, ProviderFailure
FailurePolicy                      # category, retry_allowed, counts_against_circuit, is_timeout
TRANSPORT_POLICIES, PROVIDER_POLICIES
HttpTransport, HttpxTransport, HttpRequest, HttpResponse
ModelCallFailure, ModelFailureCategory
```

`ReliableModelClient` is the only owner of retries (network failures, 429 and 5xx, with growing pauses), of the overall time
limit of a call (300 s), of cancellation and of a simple circuit breaker. It reads every decision from the policy of the
failure and never checks its class. A read timeout is retried once, at once and with the overall limit doubled
(`TIMEOUT_RETRIES = 1`, `TIMEOUT_RETRY_FACTOR = 2`: 600 s), and does not open the breaker: it is a property of the task on
this node, not of the provider, so the retry under the same limit would give the same result. The read timeout of the HTTP
client is that of the retry; the overall deadline cuts the first attempt to 300 s. Failures leave the layer as
`ProviderError` subclasses with a closed list of kinds.

## Boundaries

The layer does not build requests of a particular provider and does not parse their answers: that is the adapter
(`../provider_adapters/openai_client.py`). It does not decide the status of a point: the point executor turns a failure into
`api_error`.

Method: [Task construction](../../../../docs/03-task-construction.md), section "Statuses of a point".
