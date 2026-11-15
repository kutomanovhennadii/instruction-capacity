# exceptions — one exception hierarchy with error codes

Every error that the package raises on purpose is a subclass of `IECError` and carries a machine-readable `error_code`. A
reader needs this folder to catch the errors of one layer or to read the code of a failed command.

## What is here

| Module | Role |
|---|---|
| `base_errors.py` | the root `IECError`, its subclasses and their default error codes |

## Key signatures

```python
IECError(message: str, *, error_code: str | None = None)   # .message, .error_code
IECError.from_cause(message, cause, *, error_code=None)
ConfigError              # CONFIG_ERROR: the configuration is missing, incomplete or invalid
VocabError               # VOCAB_ERROR: building or loading the vocabulary failed
ProviderError            # PROVIDER_ERROR: a model or tokenizer adapter failed
ValidationError          # VALIDATION_ERROR: the answer validator cannot produce a result
ContractViolationError   # CONTRACT_VIOLATION: data at a module boundary violate a contract
PassportError            # PASSPORT_ERROR: the passport cannot be computed from the points
```

## Boundaries

The folder only defines the classes; every layer raises them with its own codes, listed in the README of that layer. An
admissible empty result is not an exception, and a bad answer of a model is data, not an error. `from_cause()` keeps the
original cause in `__cause__`; the caller decides when to raise.
