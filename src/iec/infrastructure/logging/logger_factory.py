"""Logger factory and context adapter of the IEC bench.

The module uses the standard library only. One marked stream handler belongs to the root logger
of the package, and child loggers get it through ordinary propagation; a repeated configuration
adds no second handler. The level and the log file are set at the setup of a command, by the
observability builder (``iec.infrastructure.observability``) or by the caller. ``get_logger``
configures the package with the default level on the first use only and never resets a level
that was set.

A file handler writes every record to disk at once, without write buffering, so a long run can
be followed with ``tail`` while the process works.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from iec.infrastructure.logging.log_context import LogContext

PACKAGE_LOGGER_NAME = "iec"
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

#: Marks of the handlers the package attached itself: a repeated configuration finds them.
_HANDLER_MARKER = "_iec_default_handler"
_FILE_HANDLER_MARKER = "_iec_file_handler"


class LogContextAdapter(logging.LoggerAdapter):
    """Prepends the rendered ``LogContext`` to every log message."""

    def __init__(self, logger: logging.Logger, context: LogContext) -> None:
        """Keep the logger and the context of its records."""
        super().__init__(logger, {})
        self.context = context

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Add the context prefix without changing the record parameters."""
        return f"{self.context.render()} {msg}", kwargs


class _FlushingFileHandler(logging.FileHandler):
    """A file handler that flushes every record to disk at once.

    The ordinary ``FileHandler`` relies on stream buffering: during a long run, until the buffer
    is full, the file looks empty - exactly the case the file log exists for. Here every record
    is flushed immediately, so ``tail`` shows an event at the moment it happens.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write the event and flush the stream buffer immediately."""
        super().emit(record)
        self.flush()


def configure_package_logger(*, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Set the level of the package root logger and give it one stream handler.

    Args:
        level: Level of the package root logger.

    Returns:
        The configured ``logging.Logger``.

    Side effects:
        Sets the level, switches off propagation of records to the process root logger and adds
        at most one marked ``StreamHandler``: a repeated call does not duplicate the handler.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not _has_stream_handler(logger):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return logger


def attach_file_handler(path: str | Path) -> Path:
    """Attach a file handler to the package logger and return the file path.

    Args:
        path: Path of the log file; missing directories are created.

    Returns:
        Absolute path of the file the log is written to.

    Side effects:
        Configures the package logger with the default level when nobody did it before, creates
        the directory of the file, opens the file for appending and adds at most one marked
        handler per path: a repeated call with the same path does not double the records. The
        level of the package logger is not changed.
    """
    resolved = Path(path).expanduser().resolve()
    logger = _configured_package_logger()
    for handler in logger.handlers:
        if getattr(handler, _FILE_HANDLER_MARKER, None) == str(resolved):
            return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = _FlushingFileHandler(resolved, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    setattr(handler, _FILE_HANDLER_MARKER, str(resolved))
    logger.addHandler(handler)
    return resolved


def get_logger(context: LogContext) -> LogContextAdapter:
    """Return a context adapter of the child logger of the given module.

    Args:
        context: Context added to every record.

    Returns:
        An adapter ready for ``info``, ``warning`` and ``error`` calls.

    Side effects:
        Configures the package logger with the default level on the first use only.
    """
    _configured_package_logger()
    child_logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{context.module_name}")
    return LogContextAdapter(child_logger, context)


def _configured_package_logger() -> logging.Logger:
    """Return the package root logger, configured with the default level if nobody did it."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if _has_stream_handler(logger):
        return logger
    return configure_package_logger()


def _has_stream_handler(logger: logging.Logger) -> bool:
    """Tell whether the package already attached its stream handler to the logger."""
    return any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers)
