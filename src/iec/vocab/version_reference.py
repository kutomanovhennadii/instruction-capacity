"""The version reference of a vocabulary: its file, its content and its reading and writing.

A vocabulary file comes with the reference of its version, ``<vocab_id>.reference.yaml`` next to
it. The build command writes the reference once, when the version is fixed: the hash, the size,
the source, the tokenizers and the parameters of the vocabulary, the thresholds fixed in the code,
the funnel of the build and the versions of the libraries that decide the result. The verifier
compares a vocabulary file with it. This module is the one place that knows the reference: its
file name, its content and format version, the libraries it records, its reading and writing.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from iec.contracts import Vocab
from iec.contracts.format_version import check_format_version
from iec.infrastructure.configuration import load_yaml_mapping
from iec.infrastructure.exceptions import ConfigError, ContractViolationError
from iec.infrastructure.files import write_text_atomically
from iec.vocab.counting_tokenizer import CountingTokenizer
from iec.vocab.vocab_builder import PROBE_PREFIX
from iec.vocab.word_source import MIN_WORD_LENGTH

#: Ending of the name of a reference file: ``<vocab_id>.reference.yaml``.
REFERENCE_SUFFIX = ".reference.yaml"
#: Version of the format of a reference that this code writes; a reference without the field
#: is of version 1, as every published reference.
REFERENCE_FORMAT_VERSION = 1
#: Libraries whose installed versions the reference records and the verifier compares.
PROVENANCE_LIBRARIES = ("nltk", "tiktoken", "transformers", "tokenizers")

_REFERENCE_HEADER = (
    "# Reference of the vocabulary version, written by scripts/build_vocab.py right\n"
    "# after the build. A changed vocabulary is a new version with a new vocab_id.\n"
    "# Check: python scripts/verify_vocab.py --vocab-id <vocab_id>\n\n"
)


def reference_file(base_dir: Path, vocab_id: str) -> Path:
    """Return the path of the reference of a vocabulary in its folder.

    Args:
        base_dir: Folder of the vocabularies and their references.
        vocab_id: Identifier of the vocabulary.

    Returns:
        ``<base_dir>/<vocab_id>.reference.yaml``.
    """
    return base_dir / f"{vocab_id}{REFERENCE_SUFFIX}"


def installed_python() -> str:
    """Return the version of the running Python as ``major.minor.micro``."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def reference_content(vocab: Vocab, counters: Sequence[CountingTokenizer]) -> dict[str, Any]:
    """Return the content of the reference of a vocabulary just built.

    Args:
        vocab: The vocabulary just built.
        counters: The counting tokenizers of the build in the order of the config; the first
            one probed every candidate word, so its count of probes is the number of candidates.

    Returns:
        The fields of the reference in the order they are written, the format version first.
    """
    return {
        "format_version": REFERENCE_FORMAT_VERSION,
        "vocab_id": vocab.vocab_id,
        "built_at": datetime.now(UTC).date().isoformat(),
        "hash": vocab.hash,
        "size": vocab.size,
        "source": vocab.source,
        "tokenizer_names": list(vocab.tokenizer_names),
        "build_params": dict(vocab.build_params),
        "code_constants": {"min_word_length": MIN_WORD_LENGTH, "probe_prefix": PROBE_PREFIX},
        "funnel": {"candidates": counters[0].probes, "intersection": vocab.size},
        "per_tokenizer": {counter.name: counter.single for counter in counters},
        "provenance": {
            "python": installed_python(),
            **{name: version(name) for name in PROVENANCE_LIBRARIES},
        },
    }


def write_reference(
    vocab: Vocab, counters: Sequence[CountingTokenizer], base_dir: Path
) -> tuple[Path, bool]:
    """Write the reference next to the vocabulary; an existing reference is kept as it is.

    Args:
        vocab: The vocabulary just built.
        counters: The counting tokenizers of the build in the order of the config.
        base_dir: Folder of the vocabulary.

    Returns:
        The path of the reference and whether it was written now.

    Side effects:
        When the reference is absent, writes ``<base_dir>/<vocab_id>.reference.yaml`` through a
        temporary file next to it.
    """
    path = reference_file(base_dir, vocab.vocab_id)
    if path.exists():
        return path, False
    text = yaml.safe_dump(reference_content(vocab, counters), sort_keys=False, allow_unicode=True)
    write_text_atomically(path, _REFERENCE_HEADER + text)
    return path, True


def load_reference(path: Path) -> dict[str, Any]:
    """Load the reference of a vocabulary version and check its format version.

    Args:
        path: Path of the reference file.

    Returns:
        The fields of the reference as YAML gives them.

    Raises:
        ConfigError: the file is missing, unreadable, not valid YAML, not a mapping or has a
            broken format version (``VOCAB_REFERENCE_INVALID``), or it is of a newer format than
            this code reads (``FORMAT_VERSION_UNSUPPORTED``).
    """
    reference = load_yaml_mapping(
        path, error_code="VOCAB_REFERENCE_INVALID", description="vocab reference"
    )
    try:
        check_format_version(
            reference,
            REFERENCE_FORMAT_VERSION,
            invalid_code="VOCAB_REFERENCE_INVALID",
            what=f"vocab reference '{path}'",
        )
    except ContractViolationError as cause:
        raise ConfigError.from_cause(cause.message, cause, error_code=cause.error_code) from cause
    return reference
