# iec/infrastructure/logging — one logging mechanism with the context of a point

All modules of the package log through one root logger and prefix every record with the
context of the operation: the module and, when known, the model and the point. A reader needs
this folder to follow a long command or to send its log to a file.

## Contents

| Module | Role |
|---|---|
| `log_context.py` | typed context of IEC log records |
| `logger_factory.py` | logger factory and context adapter of the IEC bench |

## Key entry points

```python
LogContext(module_name: str, ...)          # optional identifiers of the operation or point
get_logger(context: LogContext, *, level=DEFAULT_LOG_LEVEL) -> LogContextAdapter
configure_package_logger(*, level=DEFAULT_LOG_LEVEL) -> logging.Logger
attach_file_handler(path: str | Path, *, level=DEFAULT_LOG_LEVEL) -> Path
LogContextAdapter, PACKAGE_LOGGER_NAME, DEFAULT_LOG_FORMAT, DEFAULT_LOG_LEVEL, LOG_FILE_ENV_VAR
```

The standard library only. Repeated configuration does not duplicate handlers. A file handler
writes every record to disk at once, so a long run can be followed with `tail`; it is switched
on by `attach_file_handler` or by the environment variable `IEC_LOG_FILE`.

## Boundaries

The folder does not decide what to log; it only gives the loggers. Telemetry of operations is
in `../telemetry`.
