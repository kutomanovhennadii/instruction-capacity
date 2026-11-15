"""Saving and loading the ``Vocab`` vocabulary in ``assets/vocab/`` with an integrity check.

A vocabulary file is JSON: the field ``format_version`` first, then the snapshot of the ``Vocab``
contract. The file is written reliably, through a temporary file next to it, and its format
version is read by the one rule of the stored formats (specification §5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iec.contracts import Vocab
from iec.contracts.format_version import check_format_version
from iec.infrastructure.exceptions import ContractViolationError, VocabError
from iec.infrastructure.files import write_text_atomically

#: Version of the format of a vocabulary file that this code writes; a file without the field
#: is of version 1, as every published vocabulary.
VOCAB_FILE_FORMAT_VERSION = 1


class VocabStore:
    """A file store of ``V`` vocabularies in JSON format.

    Every vocabulary lies in ``{base_dir}/{vocab_id}.json``. On loading the hash is
    recomputed from the words and compared with the stored one - a protection against
    accidental damage of the file. A saved identifier keeps its vocabulary: a changed
    vocabulary is a new version with a new identifier, and :meth:`save_new_version` refuses to
    overwrite it. The store writes only inside the given ``base_dir`` and does not choose the
    place of writing on its own.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize the store.

        Args:
            base_dir: Root directory of the vocabularies (usually ``assets/vocab/``).
        """
        self._base_dir = base_dir

    def _path_for(self, vocab_id: str) -> Path:
        """Return the path of the vocabulary file by its identifier."""
        return self._base_dir / f"{vocab_id}.json"

    def exists(self, vocab_id: str) -> bool:
        """Return whether a saved vocabulary with this identifier exists."""
        return self._path_for(vocab_id).is_file()

    def save(self, vocab: Vocab) -> Path:
        """Save the vocabulary to disk as human-readable JSON.

        Args:
            vocab: The vocabulary to save.

        Returns:
            Path of the written file.

        Side effects:
            Creates ``base_dir`` when missing and writes ``{base_dir}/{vocab_id}.json`` through
            a temporary file next to it, so a broken-off write never leaves a damaged file.
        """
        target = self._path_for(vocab.vocab_id)
        snapshot = {"format_version": VOCAB_FILE_FORMAT_VERSION, **vocab.as_dict()}
        payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=False)
        write_text_atomically(target, payload + "\n")
        return target

    def save_new_version(self, vocab: Vocab) -> Path:
        """Save the vocabulary unless another vocabulary is saved under the same identifier.

        Saving the same vocabulary again is allowed. A damaged existing file is not
        overwritten: its load error is raised.

        Args:
            vocab: The vocabulary to save.

        Returns:
            Path of the written file.

        Raises:
            VocabError: ``VOCAB_VERSION_CONFLICT`` - ``{vocab_id}.json`` exists with another
                hash, and a changed vocabulary needs a new ``vocab_id``; the errors of
                :meth:`load` for a damaged existing file.

        Side effects:
            Writes the file as :meth:`save` does.
        """
        if self.exists(vocab.vocab_id):
            existing = self.load(vocab.vocab_id)
            if existing.hash != vocab.hash:
                raise VocabError(
                    f"vocab '{vocab.vocab_id}' is already saved with hash {existing.hash}, "
                    f"the new build has hash {vocab.hash}: a changed vocabulary is a new "
                    "version - set a new vocab_id in the config",
                    error_code="VOCAB_VERSION_CONFLICT",
                )
        return self.save(vocab)

    def load(self, vocab_id: str) -> Vocab:
        """Load a vocabulary by its identifier and check its format and integrity.

        Args:
            vocab_id: Identifier of a previously saved vocabulary.

        Returns:
            The restored ``Vocab``.

        Raises:
            VocabError: the file is not found (``VOCAB_NOT_FOUND``), is malformed or has a
                broken format version (``VOCAB_FILE_INVALID``), is of a newer format than this
                code reads (``FORMAT_VERSION_UNSUPPORTED``) or its hash does not match the
                recomputed one (``VOCAB_HASH_MISMATCH``).
        """
        path = self._path_for(vocab_id)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as cause:
            raise VocabError.from_cause(
                f"vocab '{vocab_id}' not found at '{path}'",
                cause,
                error_code="VOCAB_NOT_FOUND",
            ) from cause

        malformed = f"vocab file for '{vocab_id}' is malformed: '{path}'"
        try:
            data: Any = json.loads(raw_text)
        except json.JSONDecodeError as cause:
            raise VocabError.from_cause(
                malformed, cause, error_code="VOCAB_FILE_INVALID"
            ) from cause
        try:
            check_format_version(
                data,
                VOCAB_FILE_FORMAT_VERSION,
                invalid_code="VOCAB_FILE_INVALID",
                what=f"vocab file '{path}'",
            )
        except ContractViolationError as cause:
            raise VocabError.from_cause(
                cause.message, cause, error_code=cause.error_code
            ) from cause
        try:
            words = frozenset(data["words"])
            stored_hash = data["hash"]
            vocab = Vocab(
                vocab_id=data["vocab_id"],
                words=words,
                hash=stored_hash,
                tokenizer_names=tuple(data["tokenizer_names"]),
                source=data["source"],
                build_params=dict(data["build_params"]),
            )
        except (KeyError, TypeError) as cause:
            raise VocabError.from_cause(
                malformed, cause, error_code="VOCAB_FILE_INVALID"
            ) from cause

        expected_hash = Vocab.compute_hash(words)
        if stored_hash != expected_hash:
            raise VocabError(
                f"vocab '{vocab_id}' failed integrity check: "
                f"stored hash {stored_hash!r} != recomputed {expected_hash!r}",
                error_code="VOCAB_HASH_MISMATCH",
            )
        return vocab
