# model_client — models and tokenizers behind one contract

The only boundary of the core with model providers and tokenization libraries. Only the
core's contracts go out; provider formats never leak into the generators, the validator or
the point executor.

## What is here

- `ModelClient` — protocol with an opaque `model_id` and
  `generate(Prompt, GenParams) -> RawOutput`.
- `Tokenizer` — protocol with a versioned `name`, `is_single_token()` and `count_tokens()`.
- `MockModelClient` / `MockTokenizer` / `build_mock_provider(ModelConfig)` — a
  deterministic echo of the prompt without network, for tests and dry runs.
- `TiktokenTokenizer` / `build_tiktoken_tokenizer(model_name)` — the OpenAI tokenizer;
  encodings ship with tiktoken, no network needed.
- `OpenAIModelClient` — adapter of OpenAI Chat Completions; `build_openai_http_client()`
  builds one HTTP client per process, reused between points.
- `ModelBuilder` / `build_default_model_builder()` — picks the factory by
  `ModelConfig.provider` (`"mock"`, `"openai"`) and returns the client and the tokenizer.
- `reliability/` — the reliable call layer: HTTP transport, one-attempt provider adapter
  and `ReliableModelClient`, the single owner of retries, the overall call deadline
  (`DEFAULT_OVERALL_DEADLINE_S = 300` s), cancellation and the circuit breaker.

Key signatures:

```python
ModelClient.generate(prompt: Prompt, params: GenParams) -> RawOutput
Tokenizer.count_tokens(text: str) -> int
build_default_model_builder() -> ModelBuilder
ModelBuilder.build(config: ModelConfig) -> tuple[ModelClient, Tokenizer]
build_tiktoken_tokenizer(model_name: str) -> TiktokenTokenizer
```

## Translation on the boundary

- the prompt goes as one user message without a system role: the prompt template is a
  protocol constant;
- `GenParams.max_tokens` is sent as `max_completion_tokens`; `seed` only when set;
- models of the gpt-5.4 family get `reasoning_effort = "none"`: capacity is measured
  without hidden reasoning;
- `usage.prompt_tokens` / `usage.completion_tokens` become `TokenUsage(prompt_tokens,
  response_tokens)`;
- `content: null` (a refusal) gives `text = ""`: a bad answer is data, not an exception;
- `raw` keeps the parsed response JSON for audit.

Retries: only what a retry can cure — 429 rate limit, 5xx and network failures, with pauses
1 -> 2 -> 4 s (a longer `Retry-After` is respected). An exhausted quota and other 4xx are
not retried. Failures leave the layer as `ProviderError` with the codes
`PROVIDER_TIMEOUT`, `PROVIDER_API_ERROR`, `PROVIDER_RESPONSE_INVALID`.

## API key

The OpenAI key is read from the environment variable `OPENAI_API_KEY`. The factory checks
it before the first call; without it the build fails with
`ConfigError(OPENAI_API_KEY_MISSING)`. The key goes only into the `Authorization` header and
is never stored in records, logs or `RawOutput.raw`.

The command `scripts/measure_point.py` first loads a `.env` file found in the current
directory or above it; a variable already set in the environment wins. The line format is
`OPENAI_API_KEY=sk-...` without spaces or quotes.

Live adapter tests call the paid API and are excluded by default; run them only
explicitly: `pytest -m network`.

## Hugging Face tokenizer

`iec.model_client.hf_tokenizer.build_hf_tokenizer(model_name)` wraps a Hugging Face
tokenizer in the `Tokenizer` contract; its name is `hf/<model>`. The given string is checked
as is, without special tokens (`add_special_tokens=False`). The first load downloads the
tokenizer files from the Hub. A load failure raises `ProviderError` with the code
`HF_LOAD_FAILED`. The adapter is not exported from the package: importing `iec.model_client`
does not load `transformers`; only the vocabulary build command imports this module.

**Claude.** Anthropic publishes no local tokenizer, so Claude is not part of the vocabulary
tokenizer set.

## Boundaries

The layer does not build the vocabulary, the rules, the text or the prompt and does not
score answers. A new provider is one adapter here plus one `provider -> factory` entry in
`build_default_model_builder()`; nothing outside the layer changes.

Method: [Task construction](../../../docs/03-task-construction.md), sections on the limits and statuses.
