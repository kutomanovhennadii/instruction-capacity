# iec/infrastructure/exceptions — one exception hierarchy with error codes

Every error that the package raises on purpose is a subclass of `IECError` and carries a
machine-readable `error_code`. A reader needs this folder to catch the errors of one layer or
to read the code of a failed command.

## Contents

| Module | Role |
|---|---|
| `base_errors.py` | one exception hierarchy of the IEC measuring bench |

## Key entry points

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

An admissible empty result is not an exception, and a bad answer of a model is data, not an
error. The original cause of a wrapped error is kept in `__cause__`.

## Boundaries

The folder only defines the classes; every layer raises them with its own codes, listed in
the README of that layer.
