"""Mirror tests of the vocabulary build configuration."""

from pathlib import Path

import pytest

from iec.infrastructure.exceptions import ConfigError
from iec.vocab import TokenizerSpec, VocabBuildConfig, load_vocab_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "config" / "vocab.yaml"

VALID_YAML = """\
vocab_id: v_test
tokenizer_specs:
  - provider: tiktoken
    name: gpt-4o
"""


def test_shipped_config_loads_without_error() -> None:
    config = load_vocab_config(SHIPPED_CONFIG)

    assert config.vocab_id == "v1_intersection"
    assert config.max_syllables == 1
    assert len(config.tokenizer_specs) >= 1


def test_defaults_are_applied() -> None:
    config = VocabBuildConfig(
        vocab_id="v_test",
        tokenizer_specs=[TokenizerSpec(provider="tiktoken", name="gpt-4o")],
    )

    assert config.source == "cmudict"
    assert config.max_syllables == 1
    assert config.ascii_lower_only is True


def test_load_valid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")

    config = load_vocab_config(path)

    assert config.vocab_id == "v_test"
    assert config.tokenizer_specs[0].provider == "tiktoken"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text(VALID_YAML + "unexpected_key: 1\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_vocab_config(path)

    assert excinfo.value.error_code == "VOCAB_CONFIG_INVALID"


def test_empty_tokenizer_specs_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text("vocab_id: v_test\ntokenizer_specs: []\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_vocab_config(path)

    assert excinfo.value.error_code == "VOCAB_CONFIG_INVALID"


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_vocab_config(tmp_path / "does_not_exist.yaml")

    assert excinfo.value.error_code == "VOCAB_CONFIG_INVALID"
    assert excinfo.value.__cause__ is not None


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "vocab.yaml"
    path.write_text("vocab_id: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_vocab_config(path)

    assert excinfo.value.error_code == "VOCAB_CONFIG_INVALID"
