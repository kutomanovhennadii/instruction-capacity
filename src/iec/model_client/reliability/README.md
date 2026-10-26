# iec/model_client/reliability — the reliable call layer of a model provider

Every call of a live model goes through this layer. It separates three concerns that a
provider adapter would otherwise mix: delivering an HTTP request, knowing the API of one
provider, and the policy of retries, time limits and cancellation. A reader needs it to know
why a point ends with the status `api_error` and how long a call may take.

## Contents

| Module | Role |
|---|---|
| `failures.py` | normalized technical failures of the reliable call layer |
| `http_transport.py` | the transport layer: HTTP delivery only, without the semantics of any provider |
| `reliable_client.py` | the reliability layer: a common shell over any provider adapter |

## Key entry points

```python
ReliableModelClient(adapter: ProviderAdapter, max_retries=3, backoff_base_s=1.0,
                    overall_deadline_s=300.0, failure_threshold=5, cooldown_s=30.0)
ReliableModelClient.generate(prompt: Prompt, params: GenParams) -> RawOutput
ReliableModelClient.cancel(reason: str) -> None
ReliableModelClient.in_flight_snapshot() -> tuple[ActiveCall, ...]
ProviderAdapter                    # protocol: provider_id, model_id, generate_once(prompt, params, context)
CallContext(call_id, deadline_monotonic)
ActiveCall                         # a call in flight: call_id, model_id, attempt, elapsed_s, remaining_s
HttpTransport                      # protocol: send(request, context) -> HttpResponse
HttpxTransport                     # the httpx implementation
HttpRequest, HttpResponse
TransportFailure, TransportFailureKind
ProviderFailure, ProviderFailureKind, RetryHint
ModelCallFailure, ModelFailureCategory
```

`ReliableModelClient` is the only owner of retries (network failures, 429 and 5xx, with
growing pauses), of the overall time limit of a call (300 s), of cancellation and of a simple
circuit breaker. A read timeout is not retried and does not open the breaker: it is a property
of the task on this node, not of the provider. Failures leave the layer as `ProviderError`
subclasses with a closed list of kinds.

## Boundaries

The layer does not build requests of a particular provider and does not parse their answers:
that is the adapter (`../openai_client.py`). It does not decide the status of a point: the
point executor (`iec/point_execution`) turns a failure into `api_error`.

Method: [Task construction](../../../../docs/03-task-construction.md), section "Statuses of a point".
