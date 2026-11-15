# model_client — models and tokenizers behind one contract

The only boundary of the core with model providers and tokenization libraries. Only the core's contracts go out; provider
formats never leak into the generators, the validator or the point executor.

## What is here

| Folder or module | Role |
|---|---|
| `model_client.py` | `ModelClient`: the protocol of a model call, an opaque `model_id` and `generate(Prompt, GenParams) -> RawOutput` |
| `tokenizer.py` | `Tokenizer`: the protocol of a tokenizer, a versioned `name`, `is_single_token()` and `count_tokens()` |
| `model_builder.py` | `ModelBuilder` and `build_default_model_builder()`: the registry `provider -> factory` of a model and its tokenizer (`"mock"`, `"openai"`) |
| `tokenizer_builder.py` | `TokenizerBuilder` and `build_default_tokenizer_builder()`: the registry `provider -> factory` of a tokenizer |
| `provider_adapters/` | the OpenAI Chat Completions adapter and the deterministic mock |
| `tokenizer_adapters/` | tiktoken, a Hugging Face tokenizer, a `tokenizer.json` file and the count of a Gemini model |
| `reliability/` | the reliable call layer: HTTP transport, the policy of failures, retries, the overall deadline, the circuit breaker |

## Key signatures

```python
ModelClient.generate(prompt: Prompt, params: GenParams) -> RawOutput
Tokenizer.count_tokens(text: str) -> int
build_default_model_builder(telemetry: TelemetryHub | None = None) -> ModelBuilder
ModelBuilder.build(config: ModelConfig) -> tuple[ModelClient, Tokenizer]
build_default_tokenizer_builder() -> TokenizerBuilder
TokenizerBuilder.build(provider: str, name: str) -> Tokenizer
```

## Telemetry

Every call of a model is an event of the block `model_call` with the model id, the token counts and the finish reason of the
provider. The hub of the command reaches the models through `build_default_model_builder(telemetry)`; a model built without
a hub writes the log line of every call only.

## Translation on the boundary

- the prompt goes as one user message without a system role: the prompt template is a protocol constant;
- `GenParams.max_tokens` is sent as `max_completion_tokens`; `seed` only when set;
- models of the gpt-5.4 family get `reasoning_effort = "none"`: capacity is measured without hidden reasoning;
- `usage.prompt_tokens` / `usage.completion_tokens` become `TokenUsage(prompt_tokens, response_tokens)`;
- `content: null` (a refusal) gives `text = ""`: a bad answer is data, not an exception;
- `raw` keeps the parsed response JSON for audit.

Retries: only what a retry can cure — 429 rate limit, 5xx and network failures, with pauses 1 -> 2 -> 4 s (a longer
`Retry-After` is respected). An exhausted quota and other 4xx are not retried. The treatment of every kind of failure is a
policy in `reliability/failures.py`. Failures leave the layer as `ProviderError` with the codes `PROVIDER_TIMEOUT`,
`PROVIDER_API_ERROR`, `PROVIDER_RESPONSE_INVALID`.

## API key

The OpenAI key is read from the environment variable `OPENAI_API_KEY`. The factory checks it before the first call; without
it the build fails with `ConfigError(OPENAI_API_KEY_MISSING)`. The key goes only into the `Authorization` header and is never
stored in records, logs or `RawOutput.raw`. Live adapter tests call the paid API and are excluded by default; run them only
explicitly: `pytest -m network`.

## Tokenizers

`TokenizerBuilder` builds a tokenizer by its provider and the name of the model: `tiktoken`, `hf` (through `transformers`),
`hfjson` (a `tokenizer.json` file through `tokenizers`) and `gemini_count` (the model counts through `countTokens`; the key
comes from `GEMINI_API_KEY`). An unknown provider fails with `ConfigError(UNKNOWN_TOKENIZER_PROVIDER)`, a missing Gemini key
with `ConfigError(GEMINI_API_KEY_MISSING)`. Importing `iec.model_client` loads neither `transformers` nor `tokenizers`: the
factories of `hf` and `hfjson` import their adapters when they are called.

**Claude.** Anthropic publishes no local tokenizer, so Claude is not part of the vocabulary tokenizer set.

## Boundaries

The layer does not build the vocabulary, the rules, the text or the prompt and does not score answers. A new provider is one
adapter in `provider_adapters/` plus one `provider -> factory` entry in `build_default_model_builder()`; a new tokenizer is
one adapter in `tokenizer_adapters/` plus one entry in `build_default_tokenizer_builder()`; nothing outside the layer changes.

Method: [Task construction](../../../docs/03-task-construction.md), sections on the limits and statuses.
