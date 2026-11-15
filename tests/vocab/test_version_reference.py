"""Mirror tests of the version reference of a vocabulary."""

from pathlib import Path

import pytest
import yaml

from iec.contracts import Vocab
from iec.infrastructure.exceptions import ConfigError
from iec.model_client import MockTokenizer
from iec.vocab import CountingTokenizer, load_reference, write_reference
from iec.vocab.version_reference import REFERENCE_FORMAT_VERSION, installed_python, reference_file

_WORDS = frozenset({"lake", "stone"})


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


def test_the_reference_lies_next_to_the_vocabulary(tmp_path: Path) -> None:
    assert reference_file(tmp_path, "v_test") == tmp_path / "v_test.reference.yaml"


def test_the_reference_records_the_counts_and_an_existing_one_is_kept(tmp_path: Path) -> None:
    counter = CountingTokenizer(MockTokenizer())
    vocab = _vocab(_WORDS)
    counter.is_single_token("lake")
    counter.is_single_token("lake stone")

    path, written = write_reference(vocab, [counter], tmp_path)
    first = path.read_text(encoding="utf-8")
    reference = yaml.safe_load(first)
    _, written_again = write_reference(vocab, [counter], tmp_path)

    assert written is True
    assert written_again is False
    assert path.read_text(encoding="utf-8") == first
    assert reference["hash"] == vocab.hash
    assert reference["funnel"] == {"candidates": 2, "intersection": vocab.size}
    assert reference["per_tokenizer"] == {counter.name: counter.single}
    assert first.startswith("# Reference of the vocabulary version")


def test_a_written_reference_carries_its_format_version_and_reads_back(tmp_path: Path) -> None:
    counter = CountingTokenizer(MockTokenizer())
    counter.is_single_token("lake")

    path, _ = write_reference(_vocab(_WORDS), [counter], tmp_path)
    reference = load_reference(path)

    assert next(iter(reference)) == "format_version"
    assert reference["format_version"] == REFERENCE_FORMAT_VERSION
    assert reference["provenance"]["python"] == installed_python()


def test_a_missing_reference_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_reference(tmp_path / "v_test.reference.yaml")

    assert excinfo.value.error_code == "VOCAB_REFERENCE_INVALID"


def test_a_reference_that_is_not_a_mapping_is_a_config_error(tmp_path: Path) -> None:
    path = tmp_path / "v_test.reference.yaml"
    path.write_text("- not: a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_reference(path)

    assert excinfo.value.error_code == "VOCAB_REFERENCE_INVALID"


def test_a_reference_of_a_newer_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v_test.reference.yaml"
    path.write_text("format_version: 2\nvocab_id: v_test\n", encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_reference(path)

    assert excinfo.value.error_code == "FORMAT_VERSION_UNSUPPORTED"


def test_a_reference_without_a_format_version_is_read(tmp_path: Path) -> None:
    path = tmp_path / "v_test.reference.yaml"
    path.write_text("vocab_id: v_test\nsize: 2\n", encoding="utf-8")

    assert load_reference(path) == {"vocab_id": "v_test", "size": 2}
