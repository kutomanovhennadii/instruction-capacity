# observability — logging and telemetry assembled from one configuration

The setup of a command reads the observability configuration and builds the telemetry hub: the level and the file of the log,
the level of every block of observation and the sinks of the events.

## What is here

| Module | Role |
|---|---|
| `observability_config.py` | the Pydantic model of the configuration and `load_observability_config` |
| `sink_registry.py` | the kinds of sinks — `jsonl`, `console`, `opentelemetry` — each with its options |
| `observability_builder.py` | `build_observability`: the log configured, the sinks built, the hub returned |

## Key signatures

```python
load_observability_config(path: Path) -> ObservabilityConfig
build_observability(config, *, known_blocks=OPEN_PART_BLOCKS, registry=None) -> TelemetryHub
default_sink_registry() -> dict[str, SinkFactory]
```

The configuration, `config/observability.yaml`:

```yaml
format_version: 1
logging:
  level: info              # debug, info, warning or error
  file: logs/run.log       # optional
default_level: info        # the level of a block that is not listed
blocks:
  model_call: info
  fit: "off"               # a bare off works too: YAML reads it as false, the model as off
sinks:
  - kind: jsonl
    options:
      path: artefacts/telemetry.jsonl
  - kind: opentelemetry    # needs pip install "iec[opentelemetry]"
    options:
      tracer_name: iec
```

## Boundaries

The blocks and the sinks are checked before the log is touched: an unknown block, an unknown kind of sink and wrong options of
a sink stop the build with `ConfigError` (`OBSERVABILITY_CONFIG_INVALID`), a missing optional dependency with
`OBSERVABILITY_DEPENDENCY_MISSING`. A new kind of sink is a new factory in the registry; the model of the configuration does
not change. The closed part passes its own blocks of observation as `known_blocks`.
