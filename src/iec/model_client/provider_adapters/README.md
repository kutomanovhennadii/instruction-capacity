# provider_adapters — the providers behind the ModelClient contract

The adapter of OpenAI Chat Completions, the deterministic mock provider and the telemetry fields of a model call.

## What is here

| Module | Role |
|---|---|
| `openai_client.py` | `OpenAIModelClient`, the live model: one long-lived `ReliableModelClient` over `OpenAIProviderAdapter`, which makes one attempt and classifies the failures of OpenAI; `build_openai_http_client()` |
| `mock_provider.py` | `MockModelClient` and `MockTokenizer`: an exact echo of the prompt without network; `build_mock_provider(config, telemetry)` |
| `call_telemetry.py` | `provider_fields(output)`: the token counts and the finish reason of a raw answer for the event of the call |

## Boundaries

Both models record every call in the block `model_call` of the hub they get through the constructor. An adapter translates the
names of its provider at the boundary; retries, the overall deadline and the circuit breaker belong to `../reliability/`.
