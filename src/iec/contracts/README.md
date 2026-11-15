# contracts — the data on the boundaries of the layers

Passive, immutable data structures exchanged by the layers, in three folders by their role. Contracts read no files and no
configs, write no logs and contain no measurement logic. The contract classes are `@dataclass(frozen=True, slots=True)` and
`PointStatus` is a string enumeration; those that are stored have `as_dict()` and, where they are read back, `from_dict()`.
Every contract is imported from the package itself: `from iec.contracts import RuleTable`.

## What is here

| Folder or module | Role |
|---|---|
| `task/` | the task of a point: `Vocab`, `Rule` and `RuleTable`, `SourceText`, `Prompt`, `Reference` |
| `language_model/` | the model and its call: `ModelConfig`, `GenParams`, `RawOutput`, `TokenUsage`, `Tariff` |
| `measurement/` | points and a measurement: `PointConditions`, `PointRecord` and `PointStatus`, `PointOutcome`, `MeasurementDescription` |
| `format_version.py` | `check_format_version`, the one rule of reading the format version of a stored contract; `FORMAT_VERSION_UNSUPPORTED` |

## Key signatures

```python
PointConditions(D: int, n: int, m: int, seed_rules: int, seed_text: int, seed_order: int = 0)
PointRecord.from_dict(raw: dict[str, Any]) -> PointRecord
MeasurementDescription.from_dict(raw: dict[str, Any]) -> MeasurementDescription
ModelConfig(model_id: str, provider: str, gen_params: GenParams, ...)
Vocab.compute_hash(words: Iterable[str]) -> str
RuleTable(rules: tuple[Rule, ...], vocab_id: str, seed_rules: int)
check_format_version(raw: object, current: int, *, invalid_code: str, what: str) -> None
PROTOCOL_VERSION, POINT_RECORD_FORMAT_VERSION, MEASUREMENT_DESCRIPTION_FORMAT_VERSION
```

## Boundaries

Contracts check only what makes a single object invalid by itself: a broken record or description, an unknown status, a
newer format. Consistency between objects — `m <= D`, a vocabulary large enough — is checked by the generators, which hold
both the conditions and the vocabulary.

Method: [Data format](../../../docs/06-data-format.md).
