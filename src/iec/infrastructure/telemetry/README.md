# telemetry — the events of working operations, their levels and their sinks

A working operation of the bench — a model call, the generation of a task, a fit — is recorded as a telemetry event: its block
of observation, the module and the operation, the outcome, the times and the identifiers of the point. A reader needs this
folder to measure where the time of a command goes and to send the events to a file, the console or an external collector.

## What is here

| Module | Role |
|---|---|
| `telemetry_events.py` | `StepTelemetryEvent`, the point context `TelemetryContext`, `point_context()` and `ProviderFields` |
| `observation_blocks.py` | the blocks of observation of the open part and the levels `debug`, `info`, `warning`, `error`, `off` |
| `telemetry_hub.py` | `TelemetryHub`: the level of every block, the log line and the sinks |
| `step_decorators.py` | `@telemetry_step` over the methods of observed objects |
| `sinks/` | the sinks behind one interface: memory, a JSONL file, the console, OpenTelemetry |

## Key signatures

```python
TelemetryHub(levels: ObservationLevels, sinks: Sequence[TelemetrySink] = ())
TelemetryHub.log_only() -> TelemetryHub
hub.observed(block, operation, *, module_name, context=None)   # a with-block; yields ObservedCall
hub.attached(sink)                                              # a with-block; a sink for a while
hub.observes(block) -> bool
telemetry_step(operation, *, block, module_name=None, context_extractor=None,
               provider_extractor=None)
point_context(*sources) -> TelemetryContext
ObservationLevels(default=ObservationLevel.INFO, blocks={...}).level_of(block)
```

An observed object keeps its hub in the attribute `telemetry`:

```python
@dataclass
class TextGenerator:
    telemetry: TelemetryHub

    @telemetry_step("text_generate", block=TASK_GENERATION,
                    context_extractor=lambda self, params: point_context(params))
    def generate(self, params: PointConditions) -> SourceText: ...
```

## Boundaries

There is no process-wide sink: the hub is built once at the setup of a command and handed to the observed objects through
their constructors. A successful call has the severity `info` and a failed one `error`; an event reaches the log and the sinks
when its severity is not below the level of its block, and at the level `off` the call is not even timed. The decorator never
swallows an exception: it records the failure and re-raises the same exception. The subject status of a point (`ok`, `na`,
`api_error`) is not part of the event. The serialized event carries `format_version` 1.
