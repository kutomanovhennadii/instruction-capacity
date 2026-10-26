"""Mirror tests of the file store of the vocabulary."""

import json
from pathlib import Path

import pytest

from iec.contracts import Vocab
from iec.infrastructure.exceptions import VocabError
from iec.vocab import VocabStore


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
