# language_model — the contracts of the language model and its call

The model configuration, the generation parameters, the raw answer, the token counter and the tariff: what crosses the
boundary of the model layer.

## What is here

| Module | Role |
|---|---|
| `model_config.py` | `ModelConfig`: `model_id`, `provider`, `gen_params`, `tokenizer_name`, the context window and the answer ceiling |
| `gen_params.py` | `GenParams`: `temperature`, `max_tokens`, `top_p`, `seed` |
| `raw_output.py` | `RawOutput`: the raw answer `text`, `usage`, `finish_reason`, `latency_s` and the audit snapshot `raw` |
| `token_usage.py` | `TokenUsage`: `prompt_tokens`, `response_tokens`; `None` means not reported, not zero |
| `tariff.py` | `Tariff`: the price per million input and output tokens and `cost_usd`, the one formula of money |

## Boundaries

Provider names of counters and parameters do not reach these contracts: the model adapters translate them at the boundary.
