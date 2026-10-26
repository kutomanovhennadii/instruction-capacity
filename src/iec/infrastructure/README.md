# infrastructure — cross-cutting mechanisms of the measurement core

The package holds the mechanisms every working layer uses: one exception hierarchy, one
logging setup and step telemetry. There is no measurement logic here.

## What is here

- `exceptions/` — the exception hierarchy. `IECError` is the common root; subclasses
  separate configuration, vocabulary, provider, validation and contract errors. Every error
  carries a stable `error_code`.
- `logging/` — standard library logging. `LogContext` renders a stable prefix with the
  point identity (`m`, `seed_rules`, `seed_text`, `seed_order`, ...), `LogContextAdapter`
  adds it to messages, `configure_package_logger()` idempotently sets up the handler of the
  `iec` package.
- `telemetry/` — `@telemetry_step` measures the duration of a working operation, logs its
  success or failure and optionally sends an event to a sink. `TelemetryCollector` keeps
  events in memory; `global_sink.py` is a process-wide sink that connects a file to the
  decorators already in place without rebinding any of them.

Key signatures:

```python
IECError(message: str, *, error_code: str | None = None)
IECError.from_cause(message: str, cause: BaseException, *, error_code: str | None = None)

LogContext(module_name: str, ..., seed_order: int | None = None)
LogContext.render() -> str
get_logger(context: LogContext, *, level: int = logging.INFO) -> LogContextAdapter

telemetry_step(operation: str, *, module_name: str | None = None,
               context_extractor: ContextExtractor | None = None,
               sink: TelemetrySink | None = None) -> Callable
set_telemetry_sink(sink: TelemetrySink) -> None
reset_telemetry_sink() -> None
```

## Boundaries

`from_cause()` builds an exception keeping the original one in `__cause__`; the caller
decides when to raise it. Valid empty results are not exceptions.

The logger factory writes no files: its only side effect is at most one stream handler of
the package logger.

The decorator never swallows exceptions: it records a `failure` event and re-raises the
original exception. An explicit sink of a decorator takes precedence: when it is given, the
global sink is not called, so an event is never written twice. The global sink is process
state; its owner removes it in `finally`.
