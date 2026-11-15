# logging — one logging mechanism with the context of a point

All modules of the package log through the root logger `iec` and prefix every record with the context of the operation: the
module, the step and, when known, the identifiers of the point. A reader needs this folder to follow a long command or to send
its log to a file.

## What is here

| Module | Role |
|---|---|
| `log_context.py` | `LogContext`: the module, the step and the fields of the record prefix |
| `logger_factory.py` | the configuration of the package logger, the file handler and the context adapter |

## Key signatures

```python
LogContext(module_name: str, step_name: str | None = None, fields: tuple[LogField, ...] = ())
LogContext.render() -> str                     # "[module=... step=... m=8 ...]"
get_logger(context: LogContext) -> LogContextAdapter
configure_package_logger(*, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger
attach_file_handler(path: str | Path) -> Path
```

## Boundaries

The standard library only. The level and the log file are set once at the setup of a command, by the observability builder
(`../observability`) or by the caller; `get_logger` configures the default level `INFO` on the first use only and never
resets a level that was set. A repeated configuration adds no second handler. A file handler writes every record to disk at
once, so a long run can be followed with `tail`. The folder does not decide what to log; the telemetry of operations is in
`../telemetry`.
