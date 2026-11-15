"""Mirror tests of the IEC logger factory."""

import json
import logging
from pathlib import Path

from iec.infrastructure.logging import (
    LogContext,
    LogContextAdapter,
    attach_file_handler,
    configure_package_logger,
    get_logger,
)


def _package_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return the handlers the package itself attached: its stream handler and file handlers.

    pytest attaches its own capture handlers to the package logger, so counting all handlers
    would test pytest, not the package.
    """
    return [
        handler
        for handler in logger.handlers
        if getattr(handler, "_iec_default_handler", False)
        or getattr(handler, "_iec_file_handler", None) is not None
    ]


def test_configure_package_logger_is_idempotent(isolated_package_logger: logging.Logger) -> None:
    first = configure_package_logger()
    second = configure_package_logger()

    assert first is isolated_package_logger
    assert second is isolated_package_logger
    assert len(_package_handlers(isolated_package_logger)) == 1
    assert isolated_package_logger.propagate is False


def test_configure_package_logger_sets_the_level(isolated_package_logger: logging.Logger) -> None:
    configure_package_logger(level=logging.DEBUG)

    assert isolated_package_logger.level == logging.DEBUG


def test_get_logger_returns_child_context_adapter(isolated_package_logger: logging.Logger) -> None:
    adapter = get_logger(LogContext(module_name="model_builder"))

    assert isinstance(adapter, LogContextAdapter)
    assert adapter.logger.name == "iec.model_builder"
    assert len(_package_handlers(isolated_package_logger)) == 1
    assert isolated_package_logger.level == logging.INFO


def test_get_logger_keeps_a_configured_level(isolated_package_logger: logging.Logger) -> None:
    """A level set by the configuration survives the loggers of other modules."""
    configure_package_logger(level=logging.WARNING)

    get_logger(LogContext(module_name="runner"))

    assert isolated_package_logger.level == logging.WARNING


def test_adapter_prefixes_message_with_context() -> None:
    logger = logging.getLogger("iec.test_adapter")
    adapter = LogContextAdapter(logger, LogContext(module_name="runner", fields=(("m", 4),)))
    kwargs: dict[str, object] = {}

    message, returned_kwargs = adapter.process("started", kwargs)

    assert message == "[module=runner m=4] started"
    assert returned_kwargs is kwargs


def test_logging_import_resolves_to_standard_library() -> None:
    """The folder iec/infrastructure/logging does not shadow the standard library module."""
    assert logging.__file__ is not None
    assert json.__file__ is not None
    assert Path(logging.__file__).resolve().parents[1] == Path(json.__file__).resolve().parents[1]


def test_file_handler_writes_each_event_immediately(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    """The event is visible in the file at once - otherwise a long run cannot be watched."""
    target = tmp_path / "series.log"
    attach_file_handler(target)
    log = get_logger(LogContext(module_name="runner"))

    log.info("phase started")

    assert target.exists()
    assert "phase started" in target.read_text(encoding="utf-8")


def test_file_handler_is_not_attached_twice_for_the_same_path(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    """Attaching the same path again does not double the records."""
    target = tmp_path / "series.log"
    attach_file_handler(target)
    attach_file_handler(target)
    log = get_logger(LogContext(module_name="runner"))

    log.info("single event")

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_file_handler_creates_missing_directories(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    """The directory of the file is created: the observer names a path, not a place."""
    target = tmp_path / "nested" / "deeper" / "series.log"

    resolved = attach_file_handler(target)

    assert resolved.parent.is_dir()


def test_file_handler_keeps_the_level(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    configure_package_logger(level=logging.DEBUG)

    attach_file_handler(tmp_path / "series.log")

    assert isolated_package_logger.level == logging.DEBUG
