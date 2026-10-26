"""Saving and loading the ``Vocab`` vocabulary in ``assets/vocab/`` with an integrity check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iec.contracts import Vocab
from iec.infrastructure.exceptions import VocabError


class VocabStore:
    """A file store of ``V`` vocabularies in JSON format.

    Every vocabulary lies in ``{base_dir}/{vocab_id}.json``. On loading the hash is
    recomputed from the words and compared with the stored one - a protection against
    accidental damage of the file. The store writes only inside the given ``base_dir`` and
    does not choose the place of writing on its own.
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
            Creates ``base_dir`` when missing and writes ``{base_dir}/{vocab_id}.json``.
        """
        self._base_dir.mkdir(parents=True, exist_ok=True)
        target = self._path_for(vocab.vocab_id)
        payload = json.dumps(vocab.as_dict(), ensure_ascii=False, indent=2, sort_keys=False)
        target.write_text(payload + "\n", encoding="utf-8")
        return target

    def load(self, vocab_id: str) -> Vocab:
        """Load a vocabulary by its identifier and check its integrity.

        Args:
            vocab_id: Identifier of a previously saved vocabulary.

        Returns:
            The restored ``Vocab``.

        Raises:
            VocabError: the file is not found, invalid or its hash does not match the
                recomputed one (``error_code`` ``VOCAB_NOT_FOUND`` / ``VOCAB_FILE_INVALID``
                / ``VOCAB_HASH_MISMATCH`` respectively).
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

        try:
            data: Any = json.loads(raw_text)
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
        except (json.JSONDecodeError, KeyError, TypeError) as cause:
            raise VocabError.from_cause(
                f"vocab file for '{vocab_id}' is malformed: '{path}'",
                cause,
                error_code="VOCAB_FILE_INVALID",
            ) from cause

        expected_hash = Vocab.compute_hash(words)
        if stored_hash != expected_hash:
            raise VocabError(
                f"vocab '{vocab_id}' failed integrity check: "
                f"stored hash {stored_hash!r} != recomputed {expected_hash!r}",
                error_code="VOCAB_HASH_MISMATCH",
            )
        return vocab
