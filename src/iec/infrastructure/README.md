# infrastructure — the cross-cutting mechanisms of the measuring bench

Every working layer uses these mechanisms: one exception hierarchy, one logging setup, the telemetry of working operations,
the loading of YAML files, the assembly of observability from one configuration and the reliable writing of files. There is
no measurement logic here.

## What is here

| Folder | Role |
|---|---|
| `exceptions/` | the root `IECError` and its subclasses, each with a machine-readable `error_code` |
| `logging/` | loggers of the `iec` package with the context of a point in every record |
| `telemetry/` | telemetry events, blocks of observation and their levels, sinks, the hub and the `@telemetry_step` decorator |
| `configuration/` | loading of a YAML file into a Pydantic model or a plain mapping, with one rule for its errors |
| `observability/` | the observability configuration, the registry of sink kinds and the builder of the hub |
| `files/` | reliable writing of files: a whole file through a temporary file and a rename, a journal by whole appended lines |

## Key signatures

```python
IECError(message: str, *, error_code: str | None = None)
IECError.from_cause(message: str, cause: BaseException, *, error_code: str | None = None)
get_logger(context: LogContext) -> LogContextAdapter
telemetry_step(operation: str, *, block: str, module_name=None, context_extractor=None,
               provider_extractor=None)
TelemetryHub(levels: ObservationLevels, sinks: Sequence[TelemetrySink] = ())
load_yaml_model(path: Path, model: type[ModelT], *, error_code: str, description: str) -> ModelT
load_yaml_mapping(path: Path, *, error_code: str, description: str) -> dict[str, Any]
load_observability_config(path: Path) -> ObservabilityConfig
build_observability(config, *, known_blocks=OPEN_PART_BLOCKS, registry=None) -> TelemetryHub
write_text_atomically(path: Path, text: str) -> None
append_line(path: Path, line: str) -> None
```

## Boundaries

The package keeps no process-wide state of its own: the telemetry hub is built once at the setup of a command and handed to
the observed objects through their constructors. The standard library `logging` keeps its loggers per process; the package
configures only its own logger `iec`. The package depends on no other package of `iec`: the layers depend on it.
