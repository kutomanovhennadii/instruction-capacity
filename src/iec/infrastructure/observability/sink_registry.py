"""Registry of the kinds of telemetry sinks: a kind named in the configuration and its factory.

A factory checks the options of its kind with its own Pydantic model and builds the sink. A new
kind of sink is a new factory and a new entry in the registry; neither the configuration model
nor the builder change (specification §2.10). The OpenTelemetry factory imports the optional
dependency only when it is called.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.telemetry.sinks import (
    ConsoleSink,
    JsonlFileSink,
    OpenTelemetrySink,
    TelemetrySink,
    standard_error,
    standard_output,
)

#: Error code of a sink whose optional dependency is not installed.
OBSERVABILITY_DEPENDENCY_MISSING = "OBSERVABILITY_DEPENDENCY_MISSING"
#: Module of the OpenTelemetry API that the OpenTelemetry factory imports.
OPENTELEMETRY_MODULE = "opentelemetry.trace"

#: Factory of one kind of sink: takes the options of the configuration, returns the sink.
SinkFactory = Callable[[Mapping[str, Any]], TelemetrySink]


class JsonlSinkOptions(BaseModel):
    """Options of the JSONL file sink: the path of its file."""

    model_config = ConfigDict(extra="forbid")

    path: Path


class ConsoleSinkOptions(BaseModel):
    """Options of the console sink: the stream it prints to."""

    model_config = ConfigDict(extra="forbid")

    stream: Literal["stdout", "stderr"] = "stderr"


class OpenTelemetrySinkOptions(BaseModel):
    """Options of the OpenTelemetry sink: the name of its tracer."""

    model_config = ConfigDict(extra="forbid")

    tracer_name: str = "iec"


#: Streams of the console sink by their names in the configuration.
_CONSOLE_STREAMS = {"stdout": standard_output, "stderr": standard_error}


def build_jsonl_sink(options: Mapping[str, Any]) -> TelemetrySink:
    """Build the JSONL file sink.

    Raises:
        pydantic.ValidationError: the options do not satisfy ``JsonlSinkOptions``.
    """
    return JsonlFileSink(JsonlSinkOptions.model_validate(options).path)


def build_console_sink(options: Mapping[str, Any]) -> TelemetrySink:
    """Build the console sink printing to the named stream.

    Raises:
        pydantic.ValidationError: the options do not satisfy ``ConsoleSinkOptions``.
    """
    settings = ConsoleSinkOptions.model_validate(options)
    return ConsoleSink(_CONSOLE_STREAMS[settings.stream])


def build_opentelemetry_sink(options: Mapping[str, Any]) -> TelemetrySink:
    """Build the OpenTelemetry sink over a tracer of the OpenTelemetry API.

    Raises:
        pydantic.ValidationError: the options do not satisfy ``OpenTelemetrySinkOptions``.
        ConfigError: the optional dependency ``opentelemetry-api`` is not installed
            (``error_code="OBSERVABILITY_DEPENDENCY_MISSING"``).
    """
    settings = OpenTelemetrySinkOptions.model_validate(options)
    try:
        trace = import_module(OPENTELEMETRY_MODULE)
    except ModuleNotFoundError as cause:
        raise ConfigError.from_cause(
            "the opentelemetry sink needs the optional dependency opentelemetry-api: "
            "pip install 'iec[opentelemetry]'",
            cause,
            error_code=OBSERVABILITY_DEPENDENCY_MISSING,
        ) from cause

    def error_status(message: str) -> object:
        """Make the error status of a span in the OpenTelemetry API."""
        return trace.Status(trace.StatusCode.ERROR, message)

    return OpenTelemetrySink(trace.get_tracer(settings.tracer_name), error_status)


def default_sink_registry() -> dict[str, SinkFactory]:
    """Return the kinds of sinks of the open part: ``jsonl``, ``console`` and ``opentelemetry``."""
    return {
        "jsonl": build_jsonl_sink,
        "console": build_console_sink,
        "opentelemetry": build_opentelemetry_sink,
    }
