"""Mirror tests of the loader of YAML files."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from iec.infrastructure.configuration import load_yaml_mapping, load_yaml_model
from iec.infrastructure.exceptions import ConfigError

_CODE = "SAMPLE_CONFIG_INVALID"


class SampleConfig(BaseModel):
    """A configuration of one required name."""

    model_config = ConfigDict(extra="forbid")

    name: str


def _load(path: Path) -> SampleConfig:
    return load_yaml_model(path, SampleConfig, error_code=_CODE, description="sample config")


def _load_mapping(path: Path) -> dict[str, object]:
    return load_yaml_mapping(path, error_code=_CODE, description="sample file")


def test_valid_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("name: probe\n", encoding="utf-8")

    assert _load(path) == SampleConfig(name="probe")


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _load(tmp_path / "absent.yaml")

    assert caught.value.error_code == _CODE
    assert "sample config not found or unreadable" in caught.value.message


def test_invalid_yaml_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        _load(path)

    assert "sample config is not valid YAML" in caught.value.message


def test_schema_violation_keeps_its_cause(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("name: probe\ncolour: red\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        _load(path)

    assert "sample config does not satisfy the schema" in caught.value.message
    assert isinstance(caught.value.__cause__, ValidationError)


def test_a_mapping_is_loaded_as_it_is(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("name: probe\nsize: 3\n", encoding="utf-8")

    assert _load_mapping(path) == {"name": "probe", "size": 3}


def test_a_missing_mapping_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        _load_mapping(tmp_path / "absent.yaml")

    assert caught.value.error_code == _CODE
    assert "sample file not found or unreadable" in caught.value.message


def test_invalid_yaml_of_a_mapping_file_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("name: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        _load_mapping(path)

    assert "sample file is not valid YAML" in caught.value.message


def test_a_top_level_that_is_not_a_mapping_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.yaml"
    path.write_text("- not: a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        _load_mapping(path)

    assert caught.value.error_code == _CODE
    assert "sample file must be a mapping" in caught.value.message
