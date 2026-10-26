"""Logger factory and context adapter of the IEC bench.

The module uses the standard library only. One marked handler belongs to the root logger
of the package, and child loggers get it through ordinary propagation. Repeated
configuration does not create duplicate handlers.

Observing a long run: besides the stream handler, the package root logger
accepts an optional file handler - ``attach_file_handler``. It writes the same event
stream to a file immediately (without write buffering), so a heavy run or a live series
can be followed with ``tail`` while the process is running. The path is taken from the
environment variable ``IEC_LOG_FILE`` when it is set - this way observation is switched
on from outside without changing the calling modules. The handler is marked and
idempotent too: a repeated configuration for the same path does not create a second one.
"""

from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from iec.infrastructure.logging.log_context import LogContext

PACKAGE_LOGGER_NAME = "iec"
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LOG_LEVEL = logging.INFO

#: Environment variable with the path of the log file: when it is set, the file handler
#: attaches itself at the first logging configuration in the process.
LOG_FILE_ENV_VAR = "IEC_LOG_FILE"

_HANDLER_MARKER = "_iec_default_handler"
_FILE_HANDLER_MARKER = "_iec_file_handler"


class LogContextAdapter(logging.LoggerAdapter):
    """Prepends the rendered ``LogContext`` to every log message."""

    def __init__(self, logger: logging.Logger, context: LogContext) -> None:
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

    The ordinary ``FileHandler`` relies on stream buffering: during a long run, until the
    buffer is full, the file looks empty - exactly the case the file log exists for.
    Here every record is flushed immediately, so ``tail`` shows an event at the moment it
    happens.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Write the event and flush the stream buffer immediately."""
        super().emit(record)
        self.flush()


def attach_file_handler(path: str | Path, *, level: int = DEFAULT_LOG_LEVEL) -> Path:
    """Attach a file handler to the package logger and return the file path.

    Args:
        path: Path of the log file; missing directories are created.
        level: Level of the package root logger.

    Returns:
        Absolute path of the file the log is written to.

    Side effects:
        Creates the directory of the file, opens the file for appending and adds at most
        one marked handler per path: a repeated call with the same path does not create
        a second handler (otherwise every record would be doubled).
    """
    resolved = Path(path).expanduser().resolve()
    # Logging is not configured from here: ``configure_package_logger`` itself calls this
    # function when ``IEC_LOG_FILE`` is set, and the mutual call would recurse (the handler
    # checked for repetition is added only further down). The level and the stream
    # handler are set by this moment - either by the call from
    # ``configure_package_logger`` or by ``get_logger`` before the first record.
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers:
        if getattr(handler, _FILE_HANDLER_MARKER, None) == str(resolved):
            return resolved
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = _FlushingFileHandler(resolved, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    setattr(handler, _FILE_HANDLER_MARKER, str(resolved))
    logger.addHandler(handler)
    return resolved


def configure_package_logger(*, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Idempotently configure and return the root logger of the IEC package.

    Args:
        level: Level of the package root logger.

    Returns:
        The configured ``logging.Logger``.

    Side effects:
        Adds at most one ``StreamHandler``, sets the level and switches off propagation
        of records to the process root logger. When the environment variable
        ``IEC_LOG_FILE`` is set, also attaches a file handler for its path - observation
        of a long run is switched on from outside, without changes in the calling
        modules.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    env_path = os.environ.get(LOG_FILE_ENV_VAR, "").strip()
    if env_path and not any(
        getattr(handler, _FILE_HANDLER_MARKER, None) is not None for handler in logger.handlers
    ):
        attach_file_handler(env_path, level=level)
    return logger


def get_logger(context: LogContext, *, level: int = DEFAULT_LOG_LEVEL) -> LogContextAdapter:
    """Return a context adapter of the child logger for the given module.

    Args:
        context: Context added to every record.
        level: Level of the package root logger.

    Returns:
        An adapter ready for ``info``, ``warning`` and ``error`` calls.

    Side effects:
        Ensures the root logger is configured through ``configure_package_logger``.
    """
    configure_package_logger(level=level)
    child_logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{context.module_name}")
    return LogContextAdapter(child_logger, context)
