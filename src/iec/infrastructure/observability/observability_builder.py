"""Builder of observability: sets up logging and assembles the telemetry hub from a configuration.

The setup of a command calls the builder once; the hub it returns is handed to the observed
objects through their constructors (specification §2.4: the setup apart from the working cycle).
The blocks and the sinks are checked and built before the log is configured, so a wrong
configuration stops the build with a ``ConfigError`` and leaves the log as it was.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from pydantic import ValidationError

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.logging import attach_file_handler, configure_package_logger
from iec.infrastructure.observability.observability_config import (
    OBSERVABILITY_CONFIG_INVALID,
    ObservabilityConfig,
    SinkSpec,
)
from iec.infrastructure.observability.sink_registry import SinkFactory, default_sink_registry
from iec.infrastructure.telemetry import (
    OPEN_PART_BLOCKS,
    ObservationLevels,
    TelemetryHub,
    TelemetrySink,
)

#: Levels of the package log by their names in the configuration.
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def build_observability(
    config: ObservabilityConfig,
    *,
    known_blocks: frozenset[str] = OPEN_PART_BLOCKS,
    registry: Mapping[str, SinkFactory] | None = None,
) -> TelemetryHub:
    """Set up the package log and build the telemetry hub of a command.

    Args:
        config: The validated observability configuration.
        known_blocks: Names of the blocks of observation the command has; a block of the
            configuration outside this set is an error, so a misspelt name does not pass.
        registry: Kinds of sinks; the default registry of the open part when not given.

    Returns:
        The telemetry hub with the levels of the blocks and the built sinks.

    Raises:
        ConfigError: an unknown block, an unknown kind of sink or wrong options of a sink
            (``error_code="OBSERVABILITY_CONFIG_INVALID"``), or a missing optional dependency of
            a sink (``error_code="OBSERVABILITY_DEPENDENCY_MISSING"``).

    Side effects:
        Configures the package logger: its level and, when the configuration names a file, a
        file handler. Building a JSONL sink creates the directory of its file.
    """
    unknown = sorted(set(config.blocks) - known_blocks)
    if unknown:
        raise ConfigError(
            f"unknown observation blocks {unknown}; known blocks: {sorted(known_blocks)}",
            error_code=OBSERVABILITY_CONFIG_INVALID,
        )
    factories = registry if registry is not None else default_sink_registry()
    sinks = [_built_sink(spec, factories) for spec in config.sinks]
    configure_package_logger(level=_LOG_LEVELS[config.logging.level])
    if config.logging.file is not None:
        attach_file_handler(config.logging.file)
    levels = ObservationLevels(default=config.default_level, blocks=dict(config.blocks))
    return TelemetryHub(levels, sinks)


def _built_sink(spec: SinkSpec, factories: Mapping[str, SinkFactory]) -> TelemetrySink:
    """Build one sink of the configuration by its kind.

    Raises:
        ConfigError: the kind is unknown or the options do not satisfy the schema of the kind.
    """
    factory = factories.get(spec.kind)
    if factory is None:
        raise ConfigError(
            f"unknown telemetry sink kind '{spec.kind}'; known kinds: {sorted(factories)}",
            error_code=OBSERVABILITY_CONFIG_INVALID,
        )
    try:
        return factory(spec.options)
    except ValidationError as cause:
        raise ConfigError.from_cause(
            f"options of the telemetry sink '{spec.kind}' do not satisfy its schema",
            cause,
            error_code=OBSERVABILITY_CONFIG_INVALID,
        ) from cause
