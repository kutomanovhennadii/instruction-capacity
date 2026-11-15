# measurement — the contracts of points and of a measurement

The conditions of a point, its published record, the outcome of executing it and the description of a measurement.

## What is here

| Module | Role |
|---|---|
| `point_conditions.py` | `PointConditions`: the six numbers that determine the task of a point; `v = D * n`, `identity` |
| `point_record.py` | `PointRecord` and `PointStatus` (`ok`, `na`, `api_error`); `PROTOCOL_VERSION`, the version of the measurement protocol; `POINT_RECORD_FORMAT_VERSION` |
| `point_outcome.py` | `PointOutcome`: the record plus everything built on the way; only the record is published |
| `measurement_description.py` | `MeasurementDescription`: model, protocol version, vocabulary, tariff, file hashes; `MEASUREMENT_DESCRIPTION_FORMAT_VERSION` |

## Boundaries

Reading is tolerant to unknown keys. A record and a description carry `format_version`, each of its own format: one without
the field reads as version 1, one of a newer format raises `ContractViolationError` with `FORMAT_VERSION_UNSUPPORTED`. A broken
record raises `POINT_RECORD_INVALID`, a broken description `MEASUREMENT_DESCRIPTION_INVALID`.
