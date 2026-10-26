"""Mirror tests of verifying a vocabulary against its version reference.

The vocabulary and the reference are created in a temporary folder, so the tests depend
neither on the network nor on the artifact in ``assets/vocab``.
"""

from pathlib import Path

import pytest
import yaml

from iec.contracts import Vocab
from iec.infrastructure.exceptions import ConfigError, VocabError
from iec.vocab import VocabStore, VocabVerifier

WORDS = frozenset({"lake", "stone", "sword"})
VOCAB_ID = "v_test"


def _vocab(words: frozenset[str] = WORDS) -> Vocab:
    """Build a vocabulary with a correct hash."""
    return Vocab(
        vocab_id=VOCAB_ID,
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("mock/a", "mock/b"),
        source="cmudict_monosyllabic_singular_nouns",
        build_params={"max_syllables": 1, "ascii_lower_only": True},
    )


def _reference(vocab: Vocab, **overrides: object) -> dict[str, object]:
    """Build a reference matching the vocabulary, with optional field overrides."""
    data: dict[str, object] = {
        "vocab_id": vocab.vocab_id,
        "built_at": "2026-08-02",
        "hash": vocab.hash,
        "size": vocab.size,
        "source": vocab.source,
        "tokenizer_names": list(vocab.tokenizer_names),
        "build_params": dict(vocab.build_params),
    }
    data.update(overrides)
    return data


def _prepare(tmp_path: Path, vocab: Vocab, reference: dict[str, object]) -> VocabVerifier:
    """Write the vocabulary and the reference to disk and return a ready verifier."""
    VocabStore(tmp_path).save(vocab)
    (tmp_path / f"{vocab.vocab_id}.reference.yaml").write_text(
        yaml.safe_dump(reference, allow_unicode=True), encoding="utf-8"
    )
    return VocabVerifier(tmp_path)


def test_matching_vocab_passes(tmp_path: Path) -> None:
    vocab = _vocab()
    verifier = _prepare(tmp_path, vocab, _reference(vocab))

    result = verifier.verify(VOCAB_ID, check_provenance=False)

    assert result.is_valid is True
    assert result.mismatches == ()
    assert result.vocab.words == WORDS
    assert result.reference_built_at == "2026-08-02"


def test_changed_words_are_detected_even_when_file_is_self_consistent(tmp_path: Path) -> None:
    # The key scenario: the vocabulary is rebuilt, the file is internally consistent (the
    # hash is recomputed), but the version is not the same. The integrity check lets it
    # through, the comparison with the reference does not.
    reference_vocab = _vocab()
    reference = _reference(reference_vocab)
    tampered = _vocab(frozenset({"lake", "stone"}))
    verifier = _prepare(tmp_path, tampered, reference)

    result = verifier.verify(VOCAB_ID, check_provenance=False)

    assert result.is_valid is False
    assert any("hash mismatch" in problem for problem in result.mismatches)
    assert any("size mismatch" in problem for problem in result.mismatches)


def test_corrupted_file_fails_integrity_check(tmp_path: Path) -> None:
    vocab = _vocab()
    verifier = _prepare(tmp_path, vocab, _reference(vocab))
    # Damage the words without recomputing the hash: the mismatch must surface on loading.
    path = tmp_path / f"{VOCAB_ID}.json"
    path.write_text(path.read_text(encoding="utf-8").replace("lake", "lakx"), encoding="utf-8")

    with pytest.raises(VocabError) as excinfo:
        verifier.verify(VOCAB_ID, check_provenance=False)

    assert excinfo.value.error_code == "VOCAB_HASH_MISMATCH"


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("tokenizer_names", ["mock/a"], "tokenizer set mismatch"),
        ("source", "other_source", "source mismatch"),
        ("build_params", {"max_syllables": 2}, "build_params mismatch"),
    ],
)
def test_reference_field_mismatches_are_reported(
    tmp_path: Path, field: str, value: object, expected_fragment: str
) -> None:
    vocab = _vocab()
    verifier = _prepare(tmp_path, vocab, _reference(vocab, **{field: value}))

    result = verifier.verify(VOCAB_ID, check_provenance=False)

    assert result.is_valid is False
    assert any(expected_fragment in problem for problem in result.mismatches)


def test_missing_reference_raises_config_error(tmp_path: Path) -> None:
    VocabStore(tmp_path).save(_vocab())

    with pytest.raises(ConfigError) as excinfo:
        VocabVerifier(tmp_path).verify(VOCAB_ID)

    assert excinfo.value.error_code == "VOCAB_REFERENCE_INVALID"


def test_malformed_reference_raises_config_error(tmp_path: Path) -> None:
    vocab = _vocab()
    VocabStore(tmp_path).save(vocab)
    (tmp_path / f"{VOCAB_ID}.reference.yaml").write_text("- not: a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        VocabVerifier(tmp_path).verify(VOCAB_ID)

    assert excinfo.value.error_code == "VOCAB_REFERENCE_INVALID"


def test_provenance_drift_does_not_invalidate_artifact(tmp_path: Path) -> None:
    # Library versions will surely not match the invented ones - but the vocabulary stays fit.
    vocab = _vocab()
    reference = _reference(vocab, provenance={"nltk": "0.0.1", "python": "3.0.0"})
    verifier = _prepare(tmp_path, vocab, reference)

    result = verifier.verify(VOCAB_ID, check_provenance=True)

    assert result.is_valid is True
    assert len(result.provenance_drift) == 2


def test_provenance_check_can_be_skipped(tmp_path: Path) -> None:
    vocab = _vocab()
    reference = _reference(vocab, provenance={"nltk": "0.0.1"})
    verifier = _prepare(tmp_path, vocab, reference)

    result = verifier.verify(VOCAB_ID, check_provenance=False)

    assert result.provenance_drift == ()


def test_require_valid_returns_vocab_when_matching(tmp_path: Path) -> None:
    vocab = _vocab()
    verifier = _prepare(tmp_path, vocab, _reference(vocab))

    assert verifier.require_valid(VOCAB_ID).words == WORDS


def test_require_valid_raises_on_mismatch(tmp_path: Path) -> None:
    # The form for the measuring code: a mismatch must be an error, not a warning.
    reference = _reference(_vocab())
    verifier = _prepare(tmp_path, _vocab(frozenset({"lake"})), reference)

    with pytest.raises(VocabError) as excinfo:
        verifier.require_valid(VOCAB_ID)

    assert excinfo.value.error_code == "VOCAB_REFERENCE_MISMATCH"
    assert "hash mismatch" in excinfo.value.message
