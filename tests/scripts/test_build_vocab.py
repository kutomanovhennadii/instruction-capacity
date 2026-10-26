"""The build command: tokenizer registry and the version guard; no corpora, no network."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from iec.contracts import Vocab
from iec.infrastructure.exceptions import ConfigError, VocabError
from iec.vocab import TokenizerSpec, VocabStore

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_vocab.py"
_WORDS = frozenset({"lake", "stone"})


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_vocab", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vocab(words: frozenset[str]) -> Vocab:
    return Vocab(
        vocab_id="v_test",
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("mock",),
        source="test",
        build_params={},
    )


def test_new_version_is_saved(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)

    path = _module().save_new_version(store, _vocab(_WORDS))

    assert path.is_file()
    assert store.load("v_test").words == _WORDS


def test_same_vocabulary_may_be_saved_again(tmp_path: Path) -> None:
    module = _module()
    store = VocabStore(tmp_path)

    module.save_new_version(store, _vocab(_WORDS))
    module.save_new_version(store, _vocab(_WORDS))

    assert store.load("v_test").words == _WORDS


def test_changed_vocabulary_under_the_same_id_is_refused(tmp_path: Path) -> None:
    module = _module()
    store = VocabStore(tmp_path)
    module.save_new_version(store, _vocab(_WORDS))

    with pytest.raises(VocabError) as excinfo:
        module.save_new_version(store, _vocab(frozenset({"lake"})))

    assert excinfo.value.error_code == "VOCAB_VERSION_CONFLICT"
    assert store.load("v_test").words == _WORDS


def test_unknown_tokenizer_provider_is_a_config_error() -> None:
    with pytest.raises(ConfigError) as excinfo:
        _module().build_tokenizers([TokenizerSpec(provider="claude", name="claude-x")])

    assert excinfo.value.error_code == "UNKNOWN_TOKENIZER_PROVIDER"
