"""Mirror tests of the vocabulary contract."""

import json
from dataclasses import FrozenInstanceError

import pytest

from iec.contracts import Vocab


def _make_vocab(words: frozenset[str]) -> Vocab:
    """Assemble a vocabulary with a hash computed by the canonical method of the contract."""
    return Vocab(
        vocab_id="v1_intersection",
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("tiktoken/gpt-4o", "hf/meta-llama/Llama-3.1-8B"),
        source="cmudict_monosyllabic",
        build_params={"max_syllables": 1, "ascii_lower_only": True},
    )


def test_fields_are_stored_as_given() -> None:
    words = frozenset({"lake", "cloud", "stone"})
    vocab = _make_vocab(words)

    assert vocab.vocab_id == "v1_intersection"
    assert vocab.words == words
    assert vocab.tokenizer_names == ("tiktoken/gpt-4o", "hf/meta-llama/Llama-3.1-8B")
    assert vocab.source == "cmudict_monosyllabic"
    assert vocab.build_params == {"max_syllables": 1, "ascii_lower_only": True}


def test_size_equals_word_count() -> None:
    vocab = _make_vocab(frozenset({"lake", "cloud", "stone"}))

    assert vocab.size == 3
    assert vocab.size == len(vocab.words)


def test_max_pairs_is_half_of_size_rounded_down() -> None:
    even = _make_vocab(frozenset({"lake", "cloud", "stone", "fish"}))
    odd = _make_vocab(frozenset({"lake", "cloud", "stone"}))

    assert even.max_pairs == 2
    assert odd.max_pairs == 1


def test_as_dict_is_json_serializable_with_sorted_words() -> None:
    vocab = _make_vocab(frozenset({"stone", "lake", "cloud"}))

    snapshot = vocab.as_dict()

    assert snapshot["words"] == ["cloud", "lake", "stone"]
    assert snapshot["tokenizer_names"] == ["tiktoken/gpt-4o", "hf/meta-llama/Llama-3.1-8B"]
    assert snapshot["size"] == 3
    # The snapshot must be writable to JSON without losses.
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_as_dict_does_not_alias_build_params() -> None:
    vocab = _make_vocab(frozenset({"lake"}))

    snapshot = vocab.as_dict()
    build_params = snapshot["build_params"]
    assert isinstance(build_params, dict)
    build_params["max_syllables"] = 99

    assert vocab.build_params == {"max_syllables": 1, "ascii_lower_only": True}


def test_hash_is_reproducible_and_order_independent() -> None:
    first = Vocab.compute_hash(frozenset({"lake", "cloud", "stone"}))
    second = Vocab.compute_hash({"stone", "cloud", "lake"})

    assert first == second
    assert first.startswith("sha256:")


def test_hash_differs_for_different_word_sets() -> None:
    with_extra = Vocab.compute_hash({"lake", "cloud", "stone"})
    without_extra = Vocab.compute_hash({"lake", "cloud"})

    assert with_extra != without_extra


def test_contract_is_frozen() -> None:
    vocab = _make_vocab(frozenset({"lake"}))

    with pytest.raises(FrozenInstanceError):
        vocab.vocab_id = "changed"  # type: ignore[misc]
