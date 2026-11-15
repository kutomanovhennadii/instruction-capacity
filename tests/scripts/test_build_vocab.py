"""The build command: tokenizers through the registry and the published configs; no network."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from iec.infrastructure.exceptions import ConfigError
from iec.model_client import MockTokenizer, TokenizerBuilder, build_default_tokenizer_builder
from iec.vocab import TokenizerSpec, VocabVerifier, load_vocab_config

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "build_vocab.py"
_ASSETS = _ROOT / "assets" / "vocab"
_CONFIGS = (_ROOT / "config" / "vocab.yaml", *sorted((_ROOT / "config" / "vocab").glob("*.yaml")))


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_vocab", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_tokenizer_provider_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _module().build_tokenizers([TokenizerSpec(provider="claude", name="claude-x")])

    assert excinfo.value.error_code == "UNKNOWN_TOKENIZER_PROVIDER"


def test_the_count_of_a_gemini_model_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigError) as excinfo:
        _module().build_tokenizers([TokenizerSpec(provider="gemini_count", name="m")])

    assert excinfo.value.error_code == "GEMINI_API_KEY_MISSING"


def test_tokenizers_are_built_by_the_registry_in_the_order_of_the_config() -> None:
    builder = TokenizerBuilder(factories={"mock": lambda name: MockTokenizer(name=f"mock/{name}")})
    specs = [TokenizerSpec(provider="mock", name="b"), TokenizerSpec(provider="mock", name="a")]

    tokenizers = _module().build_tokenizers(specs, builder)

    assert [tokenizer.name for tokenizer in tokenizers] == ["mock/b", "mock/a"]


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda path: path.stem)
def test_a_published_config_names_the_tokenizers_of_its_reference(config: Path) -> None:
    parsed = load_vocab_config(config)
    reference = yaml.safe_load(
        (_ASSETS / f"{parsed.vocab_id}.reference.yaml").read_text(encoding="utf-8")
    )

    names = [f"{spec.provider}/{spec.name}" for spec in parsed.tokenizer_specs]
    known = build_default_tokenizer_builder().known_providers()

    assert names == reference["tokenizer_names"]
    assert all(spec.provider in known for spec in parsed.tokenizer_specs)


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda path: path.stem)
def test_a_published_vocabulary_matches_its_reference(config: Path) -> None:
    vocab_id = load_vocab_config(config).vocab_id

    assert VocabVerifier(_ASSETS).verify(vocab_id, check_provenance=False).is_valid
