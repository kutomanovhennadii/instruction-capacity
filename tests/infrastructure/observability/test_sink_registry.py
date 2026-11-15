"""Mirror tests of the registry of the kinds of telemetry sinks."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.observability import sink_registry
from iec.infrastructure.observability.sink_registry import (
    OBSERVABILITY_DEPENDENCY_MISSING,
    build_console_sink,
    build_jsonl_sink,
    build_opentelemetry_sink,
    default_sink_registry,
)
from iec.infrastructure.telemetry import (
    ConsoleSink,
    JsonlFileSink,
    OpenTelemetrySink,
    StepTelemetryEvent,
    TelemetryStatus,
)

EventFactory = Callable[..., StepTelemetryEvent]


def test_default_registry_names_three_kinds() -> None:
    assert sorted(default_sink_registry()) == ["console", "jsonl", "opentelemetry"]


def test_jsonl_factory_builds_the_file_sink(tmp_path: Path) -> None:
    sink = build_jsonl_sink({"path": str(tmp_path / "telemetry.jsonl")})

    assert isinstance(sink, JsonlFileSink)
    assert sink.path == tmp_path / "telemetry.jsonl"


def test_console_factory_prints_to_the_named_stream(
    make_event: EventFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    sink = build_console_sink({"stream": "stdout"})

    sink(make_event())

    assert isinstance(sink, ConsoleSink)
    assert '"block": "model_call"' in capsys.readouterr().out


def test_unknown_option_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        build_jsonl_sink({"path": str(tmp_path / "telemetry.jsonl"), "rotate": True})


def test_missing_opentelemetry_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(name: str) -> object:
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(sink_registry, "import_module", absent)

    with pytest.raises(ConfigError) as caught:
        build_opentelemetry_sink({})

    assert caught.value.error_code == OBSERVABILITY_DEPENDENCY_MISSING


def test_opentelemetry_factory_uses_the_tracer_of_the_api(
    make_event: EventFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []
    tracer_names: list[str] = []

    class Span:
        def __init__(self, name: str) -> None:
            self.name = name

        def set_status(self, status: object, /) -> None:
            calls.append((self.name, status))

        def end(self, end_time: int | None = None) -> None:
            calls.append((self.name, "end"))

    class Tracer:
        def start_span(
            self, name: str, *, attributes: object = None, start_time: object = None
        ) -> Span:
            return Span(name)

    def get_tracer(name: str) -> Tracer:
        tracer_names.append(name)
        return Tracer()

    api = SimpleNamespace(
        get_tracer=get_tracer,
        Status=lambda code, message: ("status", code, message),
        StatusCode=SimpleNamespace(ERROR="ERROR"),
    )
    monkeypatch.setattr(sink_registry, "import_module", lambda name: api)

    sink = build_opentelemetry_sink({"tracer_name": "iec-test"})
    sink(make_event(status=TelemetryStatus.FAILURE, error_message="bad answer"))

    assert isinstance(sink, OpenTelemetrySink)
    assert tracer_names == ["iec-test"]
    assert calls == [
        ("model_call.model_generate", ("status", "ERROR", "bad answer")),
        ("model_call.model_generate", "end"),
    ]
