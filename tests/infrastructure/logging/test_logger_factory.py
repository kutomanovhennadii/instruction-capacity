"""Mirror tests of the IEC logger factory."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from iec.infrastructure.logging import (
    LOG_FILE_ENV_VAR,
    PACKAGE_LOGGER_NAME,
    LogContext,
    LogContextAdapter,
    attach_file_handler,
    configure_package_logger,
    get_logger,
)


def _package_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Return the handlers the package itself attached: its stream handler and file handlers.

    pytest attaches its own capture handlers to the package logger (pytest 9 does so for
    every non-propagating logger), so counting all handlers would test pytest, not the package.
    """
    return [
        handler
        for handler in logger.handlers
        if getattr(handler, "_iec_default_handler", False)
        or getattr(handler, "_iec_file_handler", None) is not None
    ]


@pytest.fixture
def isolated_package_logger() -> Iterator[logging.Logger]:
    """Isolate the global state of the package logger between tests."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers.clear()
    try:
        yield logger
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def test_configure_package_logger_is_idempotent(
    isolated_package_logger: logging.Logger,
) -> None:
    first = configure_package_logger()
    second = configure_package_logger()

    assert first is isolated_package_logger
    assert second is isolated_package_logger
    assert len(_package_handlers(isolated_package_logger)) == 1
    assert isolated_package_logger.propagate is False


def test_get_logger_returns_child_context_adapter(
    isolated_package_logger: logging.Logger,
) -> None:
    context = LogContext(module_name="model_builder", model_id="mock")

    adapter = get_logger(context)

    assert isinstance(adapter, LogContextAdapter)
    assert adapter.logger.name == "iec.model_builder"
    assert len(_package_handlers(isolated_package_logger)) == 1


def test_adapter_prefixes_message_with_context() -> None:
    logger = logging.getLogger("iec.test_adapter")
    adapter = LogContextAdapter(logger, LogContext(module_name="runner", m=4))
    kwargs: dict[str, object] = {}

    message, returned_kwargs = adapter.process("started", kwargs)

    assert message == "[module=runner m=4] started"
    assert returned_kwargs is kwargs


def test_logging_import_resolves_to_standard_library() -> None:
    assert logging.__file__ is not None
    assert "Repository\\IEC\\src\\logging" not in logging.__file__


def test_file_handler_writes_each_event_immediately(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    """The event is visible in the file at once - otherwise a long run cannot be watched.

    The handler exists for this property: a heavy run or a live series is followed with
    ``tail`` while the process is running, and ordinary buffering would show an empty file
    until the very end of the work.
    """
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


def test_environment_variable_attaches_the_file_handler(
    isolated_package_logger: logging.Logger,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``IEC_LOG_FILE`` switches observation on from outside, without changing the caller.

    This way a run is followed without changing anything in the test or the series
    script: the environment variable is set before the start, and the whole event stream
    of the package goes to the file.
    """
    target = tmp_path / "from_env.log"
    monkeypatch.setenv(LOG_FILE_ENV_VAR, str(target))

    get_logger(LogContext(module_name="runner")).info("event from env")

    assert target.exists()
    assert "event from env" in target.read_text(encoding="utf-8")


def test_environment_variable_is_ignored_when_empty(
    isolated_package_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty variable means no observation, not a file with an empty name."""
    monkeypatch.setenv(LOG_FILE_ENV_VAR, "   ")

    logger = configure_package_logger()

    assert len(_package_handlers(logger)) == 1
