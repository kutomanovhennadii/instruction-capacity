"""Mirror tests of the file store of the vocabulary."""

import json
from pathlib import Path

import pytest

from iec.contracts import Vocab
from iec.infrastructure.exceptions import VocabError
from iec.vocab import VocabStore
from iec.vocab.vocab_store import VOCAB_FILE_FORMAT_VERSION

_WORDS = frozenset({"lake", "stone"})


def _make_vocab(vocab_id: str = "v_test") -> Vocab:
    """Assemble a vocabulary with a consistent hash for a round trip."""
    words = frozenset({"lake", "cloud", "stone"})
    return Vocab(
        vocab_id=vocab_id,
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("tiktoken/gpt-4o", "hf/Qwen/Qwen2.5-0.5B"),
        source="cmudict_monosyllabic",
        build_params={"max_syllables": 1, "ascii_lower_only": True},
    )


def _vocab(words: frozenset[str]) -> Vocab:
    """Assemble a vocabulary of the given words under the identifier ``v_test``."""
    return Vocab(
        vocab_id="v_test",
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("mock",),
        source="test",
        build_params={},
    )


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    original = _make_vocab()

    written = store.save(original)
    assert written.is_file()

    loaded = store.load("v_test")
    assert loaded == original


def test_save_creates_missing_base_dir(tmp_path: Path) -> None:
    store = VocabStore(tmp_path / "nested" / "vocab")
    original = _make_vocab()

    written = store.save(original)

    assert written.is_file()
    assert written.parent.is_dir()


def test_save_writes_human_readable_json(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    written = store.save(_make_vocab())

    snapshot = json.loads(written.read_text(encoding="utf-8"))
    assert snapshot["words"] == ["cloud", "lake", "stone"]
    assert snapshot["size"] == 3
    # Indentation makes the file fit for an audit by eye.
    assert "\n  " in written.read_text(encoding="utf-8")


def test_exists_reflects_presence(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)

    assert store.exists("v_test") is False
    store.save(_make_vocab())
    assert store.exists("v_test") is True


def test_load_missing_file_raises_not_found(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)

    with pytest.raises(VocabError) as excinfo:
        store.load("absent")

    assert excinfo.value.error_code == "VOCAB_NOT_FOUND"


def test_load_tampered_hash_raises_mismatch(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    written = store.save(_make_vocab())

    # Damage the content without touching the stored hash -> mismatch on recomputation.
    snapshot = json.loads(written.read_text(encoding="utf-8"))
    snapshot["words"].append("tampered")
    written.write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        store.load("v_test")

    assert excinfo.value.error_code == "VOCAB_HASH_MISMATCH"


def test_load_malformed_json_raises_file_invalid(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        store.load("broken")

    assert excinfo.value.error_code == "VOCAB_FILE_INVALID"


def test_load_missing_key_raises_file_invalid(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    (tmp_path / "partial.json").write_text(json.dumps({"vocab_id": "partial"}), encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        store.load("partial")

    assert excinfo.value.error_code == "VOCAB_FILE_INVALID"


def test_save_leaves_no_temporary_file(tmp_path: Path) -> None:
    VocabStore(tmp_path).save(_make_vocab())

    assert sorted(path.name for path in tmp_path.iterdir()) == ["v_test.json"]


def test_save_writes_the_format_version_first(tmp_path: Path) -> None:
    written = VocabStore(tmp_path).save(_make_vocab())

    snapshot = json.loads(written.read_text(encoding="utf-8"))
    assert next(iter(snapshot)) == "format_version"
    assert snapshot["format_version"] == VOCAB_FILE_FORMAT_VERSION


def test_a_file_without_a_format_version_is_read(tmp_path: Path) -> None:
    original = _make_vocab()
    (tmp_path / "v_test.json").write_text(json.dumps(original.as_dict()), encoding="utf-8")

    assert VocabStore(tmp_path).load("v_test") == original


def test_a_file_of_a_newer_format_is_refused(tmp_path: Path) -> None:
    snapshot = {"format_version": VOCAB_FILE_FORMAT_VERSION + 1, **_make_vocab().as_dict()}
    (tmp_path / "v_test.json").write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        VocabStore(tmp_path).load("v_test")

    assert excinfo.value.error_code == "FORMAT_VERSION_UNSUPPORTED"


def test_a_broken_format_version_makes_the_file_invalid(tmp_path: Path) -> None:
    snapshot = {"format_version": "one", **_make_vocab().as_dict()}
    (tmp_path / "v_test.json").write_text(json.dumps(snapshot), encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        VocabStore(tmp_path).load("v_test")

    assert excinfo.value.error_code == "VOCAB_FILE_INVALID"


def test_new_version_is_saved(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)

    path = store.save_new_version(_vocab(_WORDS))

    assert path.is_file()
    assert store.load("v_test").words == _WORDS


def test_same_vocabulary_may_be_saved_again(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)

    store.save_new_version(_vocab(_WORDS))
    store.save_new_version(_vocab(_WORDS))

    assert store.load("v_test").words == _WORDS


def test_changed_vocabulary_under_the_same_id_is_refused(tmp_path: Path) -> None:
    store = VocabStore(tmp_path)
    store.save_new_version(_vocab(_WORDS))

    with pytest.raises(VocabError) as excinfo:
        store.save_new_version(_vocab(frozenset({"lake"})))

    assert excinfo.value.error_code == "VOCAB_VERSION_CONFLICT"
    assert store.load("v_test").words == _WORDS
