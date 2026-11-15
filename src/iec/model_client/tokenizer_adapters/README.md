# tokenizer_adapters — the tokenizers behind the Tokenizer contract

The tiktoken and Hugging Face adapters, a `tokenizer.json` file and the token count of a Gemini model.

## What is here

| Module | Role |
|---|---|
| `tiktoken_tokenizer.py` | `TiktokenTokenizer`, name `tiktoken/<model>`; the encodings ship with tiktoken |
| `hf_tokenizer.py` | `HFTokenizer`, name `hf/<model>`, through `transformers`; special tokens are not counted |
| `hub_tokenizer.py` | `HubTokenizer`, name `hfjson/<repository>`: the file `tokenizer.json` read by `tokenizers` |
| `gemini_count_tokenizer.py` | `GeminiCountTokenizer`, name `gemini_count/<model>`: the model counts through `countTokens`, a single token is the difference of two counts; failures `GEMINI_COUNT_FAILED` and `GEMINI_COUNT_INVALID` |
| `library_loading.py` | `LibraryLoading`: the expected failures of a tokenizer library become `ProviderError` with the code of the adapter |

## Boundaries

A string is checked as is: the probe form, a leading space included, is chosen by the caller. Every adapter names the
exceptions its library raises on a missing model or an unavailable network — tiktoken `KeyError`, `ValueError` and `OSError`,
`transformers` `OSError`, `ValueError` and `ImportError`, `tokenizers` `OSError` — and they become `ProviderError` with the
codes `TIKTOKEN_LOAD_FAILED`, `HF_LOAD_FAILED` and `HUB_TOKENIZER_LOAD_FAILED`; any other exception is a defect and is not
wrapped. `hf_tokenizer` and `hub_tokenizer` are imported by path: this package does not load their libraries.
