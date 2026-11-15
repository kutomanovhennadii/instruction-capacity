"""Mirror tests of the observability configuration and its loading."""

from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.infrastructure.observability import (
    OBSERVABILITY_CONFIG_INVALID,
    ObservabilityConfig,
    load_observability_config,
)
from iec.infrastructure.telemetry import ObservationLevel

#: The reference configuration shipped with the open part.
_SHIPPED = Path(__file__).resolve().parents[3] / "config" / "observability.yaml"

_FULL = """\
format_version: 1
logging:
  level: debug
  file: logs/run.log
default_level: warning
blocks:
  fit: "off"
sinks:
  - kind: jsonl
    options:
      path: telemetry.jsonl
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "observability.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_log_at_info_without_sinks() -> None:
    config = ObservabilityConfig()

    assert config.logging.level == "info"
    assert config.logging.file is None
    assert config.default_level is ObservationLevel.INFO
    assert config.blocks == {}
    assert config.sinks == []


def test_full_configuration_is_loaded(tmp_path: Path) -> None:
    config = load_observability_config(_write(tmp_path, _FULL))

    assert config.logging.level == "debug"
    assert config.logging.file == Path("logs/run.log")
    assert config.default_level is ObservationLevel.WARNING
    assert config.blocks == {"fit": ObservationLevel.OFF}
    assert config.sinks[0].kind == "jsonl"
    assert config.sinks[0].options == {"path": "telemetry.jsonl"}


def test_unquoted_off_of_yaml_means_off(tmp_path: Path) -> None:
    """YAML 1.1 reads an unquoted off as false; the configuration reads it as the level off."""
    text = "default_level: off\nblocks:\n  fit: off\n"

    config = load_observability_config(_write(tmp_path, text))

    assert config.default_level is ObservationLevel.OFF
    assert config.blocks == {"fit": ObservationLevel.OFF}


@pytest.mark.parametrize(
    "text",
    [
        "unknown_key: 1\n",
        "format_version: 2\n",
        "blocks:\n  fit: loud\n",
        "logging:\n  level: verbose\n",
        "sinks:\n  - options: {}\n",
        "logging: [unclosed\n",
    ],
)
def test_invalid_configuration_is_a_config_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError) as caught:
        load_observability_config(_write(tmp_path, text))

    assert caught.value.error_code == OBSERVABILITY_CONFIG_INVALID


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_observability_config(tmp_path / "absent.yaml")

    assert caught.value.error_code == OBSERVABILITY_CONFIG_INVALID


def test_shipped_configuration_is_valid() -> None:
    config = load_observability_config(_SHIPPED)

    assert config.default_level is ObservationLevel.INFO
    assert config.sinks == []
