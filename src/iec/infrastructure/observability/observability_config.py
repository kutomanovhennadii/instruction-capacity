"""Configuration of observability - logging and telemetry - and its loading from YAML.

The configuration is an external boundary, so it is a Pydantic model that forbids unknown keys
(specification §7.6). It sets the level and the file of the log, the level of every block of
observation and the sinks of telemetry events. A sink is named by its kind and gets its own
options; the sink registry checks the options of each kind, so a new kind of sink does not change
this model. ``format_version`` is the version of the file format: version 1 is the only one.

YAML 1.1 reads an unquoted ``off`` as ``false``; the model takes ``false`` as the level ``off``,
so ``fit: off`` means what it says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from iec.infrastructure.configuration import load_yaml_model
from iec.infrastructure.telemetry import ObservationLevel

#: Error code of an observability configuration that cannot be used.
OBSERVABILITY_CONFIG_INVALID = "OBSERVABILITY_CONFIG_INVALID"


def _level_from_yaml(value: object) -> object:
    """Read ``false`` as the level ``off``: YAML 1.1 turns an unquoted ``off`` into ``false``."""
    return ObservationLevel.OFF if value is False else value


class LoggingSettings(BaseModel):
    """Level of the package log and an optional file it is appended to besides the console."""

    model_config = ConfigDict(extra="forbid")

    level: Literal["debug", "info", "warning", "error"] = "info"
    file: Path | None = None


class SinkSpec(BaseModel):
    """One sink of telemetry events: its kind in the sink registry and its own options."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    options: dict[str, Any] = Field(default_factory=dict)


class ObservabilityConfig(BaseModel):
    """Observability of one command: the log, the levels of the blocks and the sinks."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    default_level: ObservationLevel = ObservationLevel.INFO
    blocks: dict[str, ObservationLevel] = Field(default_factory=dict)
    sinks: list[SinkSpec] = Field(default_factory=list)

    @field_validator("default_level", mode="before")
    @classmethod
    def _default_level_from_yaml(cls, value: object) -> object:
        """Take an unquoted ``off`` of YAML as the level ``off``."""
        return _level_from_yaml(value)

    @field_validator("blocks", mode="before")
    @classmethod
    def _block_levels_from_yaml(cls, value: object) -> object:
        """Take an unquoted ``off`` of YAML as the level ``off`` in the levels of the blocks."""
        if isinstance(value, dict):
            return {name: _level_from_yaml(level) for name, level in value.items()}
        return value


def load_observability_config(path: Path) -> ObservabilityConfig:
    """Load and validate the observability configuration from a YAML file.

    Args:
        path: Path of the YAML file, for example ``config/observability.yaml``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: the file is missing or unreadable, the YAML is invalid or the schema is
            violated (``error_code="OBSERVABILITY_CONFIG_INVALID"``); the cause is kept.
    """
    return load_yaml_model(
        path,
        ObservabilityConfig,
        error_code=OBSERVABILITY_CONFIG_INVALID,
        description="observability config",
    )
