"""Verification of the vocabulary ``V`` against the fixed reference of its version.

The module answers the question "is this the data on which the published results were
obtained". It is needed not only to an operator before publishing but also to the measuring
code before a measurement: a run on a substituted or damaged vocabulary gives numbers not
comparable with the rest, so the check is part of the protocol, not a manual ritual.

The check goes in three levels, each stricter than the previous one:

1. **Integrity** - the hash is recomputed from the words of the file and compared with the
   one stored in it; catches damage and manual edits of the artifact. Done by
   ``VocabStore.load``.
2. **Match with the reference** - the hash, size, tokenizer set and parameters are compared
   with ``{vocab_id}.reference.yaml``. Catches a substituted version: a vocabulary rebuilt
   with other filters is internally consistent, but its hash differs.
3. **Environment** - the installed library versions are compared with those fixed in the
   reference. A mismatch does not make the artifact unfit, so it is a warning, not an
   error: the vocabulary stays the same, but a rebuild in such an environment may give
   another result.

The module does not perform a full vocabulary build: the check needs neither the network,
nor the NLTK corpora, nor tokenizers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from iec.contracts import Vocab
from iec.infrastructure.exceptions import ConfigError, VocabError
from iec.infrastructure.logging import LogContext, get_logger
from iec.vocab.vocab_store import VocabStore

_MODULE_NAME = "vocab_verifier"

REFERENCE_SUFFIX = ".reference.yaml"

# Reference key -> distribution name for importlib.metadata. Python is checked separately.
_CHECKED_PACKAGES = {
    "nltk": "nltk",
    "tiktoken": "tiktoken",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
}

_logger = get_logger(LogContext(module_name=_MODULE_NAME))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of verifying a vocabulary against its reference.

    ``mismatches`` - mismatches with the reference that make the artifact unfit for
    measurements. ``provenance_drift`` - mismatches of library versions: the artifact stays
    fit, but a rebuild may give another result, so the list is kept apart from the errors.
    """

    vocab: Vocab
    reference_built_at: str | None
    mismatches: tuple[str, ...]
    provenance_drift: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        """Return whether the artifact matches the reference (version drift is not counted)."""
        return not self.mismatches


def _load_reference(path: Path) -> dict[str, Any]:
    """Load the reference of the vocabulary version.

    Raises:
        ConfigError: the file is missing, unreadable or not a YAML mapping
            (``error_code="VOCAB_REFERENCE_INVALID"``).
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as cause:
        raise ConfigError.from_cause(
            f"vocab reference not found or unreadable: '{path}'",
            cause,
            error_code="VOCAB_REFERENCE_INVALID",
        ) from cause

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as cause:
        raise ConfigError.from_cause(
            f"vocab reference is not valid YAML: '{path}'",
            cause,
            error_code="VOCAB_REFERENCE_INVALID",
        ) from cause

    if not isinstance(data, dict):
        raise ConfigError(
            f"vocab reference must be a mapping: '{path}'",
            error_code="VOCAB_REFERENCE_INVALID",
        )
    return data


def _compare_with_reference(vocab: Vocab, reference: dict[str, Any]) -> list[str]:
    """Compare the vocabulary with the reference and return the list of mismatches."""
    problems: list[str] = []

    if vocab.hash != reference.get("hash"):
        problems.append(
            f"hash mismatch: artifact {vocab.hash} != reference {reference.get('hash')}"
        )
    if vocab.size != reference.get("size"):
        problems.append(
            f"size mismatch: artifact {vocab.size} != reference {reference.get('size')}"
        )
    if vocab.vocab_id != reference.get("vocab_id"):
        problems.append(
            f"vocab_id mismatch: artifact {vocab.vocab_id!r} != "
            f"reference {reference.get('vocab_id')!r}"
        )
    if vocab.source != reference.get("source"):
        problems.append(
            f"source mismatch: artifact {vocab.source!r} != reference {reference.get('source')!r}"
        )

    expected_tokenizers = tuple(reference.get("tokenizer_names") or ())
    if vocab.tokenizer_names != expected_tokenizers:
        problems.append(
            f"tokenizer set mismatch: artifact {list(vocab.tokenizer_names)} != "
            f"reference {list(expected_tokenizers)}"
        )

    expected_params = reference.get("build_params") or {}
    if vocab.build_params != expected_params:
        problems.append(
            f"build_params mismatch: artifact {vocab.build_params} != reference {expected_params}"
        )

    return problems


def _compare_provenance(reference: dict[str, Any]) -> list[str]:
    """Compare the installed library versions with those fixed in the reference."""
    provenance = reference.get("provenance") or {}
    problems: list[str] = []

    expected_python = provenance.get("python")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if expected_python is not None and str(expected_python) != actual_python:
        problems.append(f"python: installed {actual_python} != reference {expected_python}")

    for key, distribution in _CHECKED_PACKAGES.items():
        expected = provenance.get(key)
        if expected is None:
            continue
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            problems.append(f"{key}: not installed (reference {expected})")
            continue
        if actual != str(expected):
            problems.append(f"{key}: installed {actual} != reference {expected}")

    return problems


class VocabVerifier:
    """Verifies a vocabulary in the store against the reference of its version.

    The store and the folder of references are one and the same ``base_dir``: the reference
    lives next to the artifact, so the version and the evidence about it do not part when
    copied.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize the verifier.

        Args:
            base_dir: Folder of vocabularies and references (usually ``assets/vocab/``).
        """
        self._base_dir = base_dir
        self._store = VocabStore(base_dir)

    def reference_path(self, vocab_id: str) -> Path:
        """Return the path to the version reference by the vocabulary identifier."""
        return self._base_dir / f"{vocab_id}{REFERENCE_SUFFIX}"

    def verify(self, vocab_id: str, *, check_provenance: bool = True) -> VerificationResult:
        """Verify the vocabulary and return a detailed result.

        Args:
            vocab_id: Identifier of the vocabulary under check.
            check_provenance: Whether to compare library versions with the reference.

        Returns:
            The verification result; mismatches are listed in the result fields, no
            exception is raised on them - the caller decides how to react (see
            :meth:`require_valid`).

        Raises:
            ConfigError: the reference is missing or invalid.
            VocabError: the vocabulary is not found, damaged or failed the integrity check.
        """
        reference = _load_reference(self.reference_path(vocab_id))
        vocab = self._store.load(vocab_id)  # level 1: integrity by the hash

        mismatches = _compare_with_reference(vocab, reference)
        drift = _compare_provenance(reference) if check_provenance else []

        if mismatches:
            _logger.error(
                f"vocab '{vocab_id}' does not match its reference: {len(mismatches)} mismatch(es)"
            )
        else:
            _logger.info(f"vocab '{vocab_id}' matches its reference (hash {vocab.hash})")
        if drift:
            _logger.warning(
                f"library versions differ from the reference for '{vocab_id}': "
                f"{len(drift)} item(s); artifact is still valid"
            )

        built_at = reference.get("built_at")
        return VerificationResult(
            vocab=vocab,
            reference_built_at=str(built_at) if built_at is not None else None,
            mismatches=tuple(mismatches),
            provenance_drift=tuple(drift),
        )

    def require_valid(self, vocab_id: str, *, check_provenance: bool = False) -> Vocab:
        """Return the vocabulary after making sure it matches its reference.

        The form for the measuring code: measurements on a vocabulary that does not match
        the reference give numbers not comparable with the rest of the measurement, so a
        mismatch here is an error, not a warning. Library version drift is not checked by
        default: it does not affect an already built artifact.

        Args:
            vocab_id: Vocabulary identifier.
            check_provenance: Whether to compare library versions (does not affect the
                fitness of the artifact).

        Returns:
            The verified vocabulary.

        Raises:
            VocabError: the artifact does not match the reference
                (``error_code="VOCAB_REFERENCE_MISMATCH"``); the list of mismatches is in
                the message.
            ConfigError: the reference is missing or invalid.
        """
        result = self.verify(vocab_id, check_provenance=check_provenance)
        if not result.is_valid:
            details = "; ".join(result.mismatches)
            raise VocabError(
                f"vocab '{vocab_id}' does not match its reference version: {details}",
                error_code="VOCAB_REFERENCE_MISMATCH",
            )
        return result.vocab
