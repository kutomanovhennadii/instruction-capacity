"""Mirror tests of the observability builder."""

import logging
from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.observability import (
    OBSERVABILITY_CONFIG_INVALID,
    ObservabilityConfig,
    build_observability,
)
from iec.infrastructure.telemetry import MODEL_CALL, MemorySink


def _config(**fields: object) -> ObservabilityConfig:
    return ObservabilityConfig.model_validate(fields)


def test_hub_gets_the_levels_and_the_sinks_of_the_configuration(
    isolated_package_logger: logging.Logger,
) -> None:
    memory = MemorySink()
    config = _config(
        blocks={"model_call": "off"}, default_level="warning", sinks=[{"kind": "memory"}]
    )

    hub = build_observability(config, registry={"memory": lambda options: memory})

    assert hub.observes(MODEL_CALL) is False
    assert hub.observes("fit") is True
    assert hub.sinks == (memory,)


def test_logging_gets_the_level_and_the_file(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    log_file = tmp_path / "logs" / "run.log"

    build_observability(_config(logging={"level": "debug", "file": str(log_file)}))

    assert isolated_package_logger.level == logging.DEBUG
    assert log_file.parent.is_dir()


def test_unknown_block_is_a_config_error(isolated_package_logger: logging.Logger) -> None:
    with pytest.raises(ConfigError) as caught:
        build_observability(_config(blocks={"modle_call": "info"}))

    assert caught.value.error_code == OBSERVABILITY_CONFIG_INVALID
    assert "modle_call" in caught.value.message


def test_blocks_of_the_closed_part_pass_with_their_known_set(
    isolated_package_logger: logging.Logger,
) -> None:
    hub = build_observability(
        _config(blocks={"batch_selection": "off"}), known_blocks=frozenset({"batch_selection"})
    )

    assert hub.observes("batch_selection") is False


def test_unknown_sink_kind_is_a_config_error(isolated_package_logger: logging.Logger) -> None:
    with pytest.raises(ConfigError) as caught:
        build_observability(_config(sinks=[{"kind": "kafka"}]))

    assert caught.value.error_code == OBSERVABILITY_CONFIG_INVALID


def test_wrong_options_of_a_sink_are_a_config_error(
    isolated_package_logger: logging.Logger, tmp_path: Path
) -> None:
    config = _config(sinks=[{"kind": "jsonl", "options": {"file": str(tmp_path / "t.jsonl")}}])

    with pytest.raises(ConfigError) as caught:
        build_observability(config)

    assert caught.value.error_code == OBSERVABILITY_CONFIG_INVALID


def test_a_wrong_configuration_leaves_the_log_as_it_was(
    isolated_package_logger: logging.Logger,
) -> None:
    """The configuration is checked before the log is touched."""
    isolated_package_logger.setLevel(logging.WARNING)

    with pytest.raises(ConfigError):
        build_observability(_config(logging={"level": "debug"}, sinks=[{"kind": "kafka"}]))

    assert isolated_package_logger.level == logging.WARNING
