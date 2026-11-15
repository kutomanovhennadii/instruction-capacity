"""Assembly of observability - logging and telemetry - from one configuration."""

from iec.infrastructure.observability.observability_builder import build_observability
from iec.infrastructure.observability.observability_config import (
    OBSERVABILITY_CONFIG_INVALID,
    LoggingSettings,
    ObservabilityConfig,
    SinkSpec,
    load_observability_config,
)
from iec.infrastructure.observability.sink_registry import (
    OBSERVABILITY_DEPENDENCY_MISSING,
    SinkFactory,
    build_console_sink,
    build_jsonl_sink,
    build_opentelemetry_sink,
    default_sink_registry,
)

__all__ = [
    "OBSERVABILITY_CONFIG_INVALID",
    "OBSERVABILITY_DEPENDENCY_MISSING",
    "LoggingSettings",
    "ObservabilityConfig",
    "SinkFactory",
    "SinkSpec",
    "build_console_sink",
    "build_jsonl_sink",
    "build_observability",
    "build_opentelemetry_sink",
    "default_sink_registry",
    "load_observability_config",
]
