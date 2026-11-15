# sinks — where telemetry events go

A sink takes every event that passed the level of its block. Any callable that takes a `StepTelemetryEvent` is a sink; the
interface is `TelemetrySink`.

## What is here

| Module | Role |
|---|---|
| `sink_contract.py` | the `TelemetrySink` interface and `OneTimeWarning` for write failures |
| `memory_sink.py` | `MemorySink`: the events of a run in memory, for tests and for counting inside a run |
| `jsonl_file_sink.py` | `JsonlFileSink`: one JSON line per event, appended to a file |
| `console_sink.py` | `ConsoleSink`: one JSON line per event to the standard error or output |
| `opentelemetry_sink.py` | `OpenTelemetrySink`: one finished span per event for external collectors |

## Key signatures

```python
class TelemetrySink(Protocol):
    def __call__(self, event: StepTelemetryEvent, /) -> None: ...
JsonlFileSink(path: Path)
ConsoleSink(stream: Callable[[], TextIO] = standard_error)
OpenTelemetrySink(tracer: SpanTracer, error_status: Callable[[str], object])
span_attributes(event) -> dict[str, str | int | float | bool]     # "iec.<field>"
```

## Boundaries

A sink handles its own write errors: it drops the event and logs one warning, because telemetry never fails the measured
operation. The JSONL file only grows; it is opened for every event, so no handle stays open. OpenTelemetry is an optional
dependency (`pip install "iec[opentelemetry]"`): this folder never imports it, and the sink works with any tracer that has
`start_span`. The application configures the OpenTelemetry SDK and its exporter; without them the API records nothing.
