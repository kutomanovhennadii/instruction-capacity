# iec/infrastructure/telemetry — telemetry events of working operations

A decorator records every call of a working operation (building a task, calling a model,
validating an answer): its duration, its outcome and the identifiers of the point. A reader
needs this folder to measure where the time of a command goes.

## Contents

| Module | Role |
|---|---|
| `telemetry_events.py` | contracts of the telemetry event and of the in-memory collector |
| `step_decorators.py` | telemetry decorator of the working operations of the IEC bench |
| `global_sink.py` | process-wide telemetry sink: attaching a file without rebinding decorators |

## Key entry points

```python
telemetry_step(operation: str, *, module_name=None, context_extractor=None, sink=None)
set_telemetry_sink(sink: TelemetrySink) -> None
reset_telemetry_sink() -> None
get_telemetry_sink() -> TelemetrySink | None
StepTelemetryEvent, TelemetryContext, TelemetryStatus, TelemetryCollector
TelemetrySink, ContextExtractor
```

An event is logged and passed to the explicit sink of the decorator or to the process-wide
sink. The outcome is `success` or `failure` by the presence of an exception; the subject
status of a point (`ok`, `na`, `api_error`) is not part of the event.

## Boundaries

The folder does not store events anywhere by itself: the sink decides. Logging is in
`../logging`.
