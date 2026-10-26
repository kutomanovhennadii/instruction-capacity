"""Vocabulary of monosyllabic single-token words and its version/hash."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vocab:
    """Result of building the vocabulary ``V`` for the IEC replacement protocol.

    A passive structure: the words are stored in canonical form (lower-case ASCII, without
    a leading space). ``hash`` is computed by the vocabulary builder with the ready
    function :meth:`compute_hash` and passed here - the contract only carries the value and
    lets it be checked on loading. ``tokenizer_names`` records the set of tokenizers whose
    intersection was taken, and ``build_params`` the filtering parameters for
    reproducibility.
    """

    vocab_id: str
    words: frozenset[str]
    hash: str
    tokenizer_names: tuple[str, ...]
    source: str
    build_params: dict[str, object]

    @property
    def size(self) -> int:
        """Return the number of words in the vocabulary."""
        return len(self.words)

    @property
    def max_pairs(self) -> int:
        """Return the maximum number of disjoint replacement pairs (= ``size // 2``).

        The upper bound of the number of rules ``m`` of one task: a rule needs two distinct
        words (the target word and its replacement), and the groups of targets and
        replacements do not intersect. It is reachable only with a lexicon without
        background words (``D = m``); the actual feasibility of a point (``D + m <= size``)
        is checked by the generators.
        """
        return self.size // 2

    @staticmethod
    def compute_hash(words: Iterable[str]) -> str:
        """Return a deterministic hash of the word set in the form ``sha256:<hex>``.

        Args:
            words: Words of the vocabulary in any order.

        Returns:
            A hash of the sorted word list; the same for the same set regardless of the
            order of enumeration. It serves to check integrity on loading (``vocab_store``)
            and as a reproducible identifier of the content.
        """
        canonical = json.dumps(sorted(words), ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the vocabulary with a sorted word list."""
        return {
            "vocab_id": self.vocab_id,
            "words": sorted(self.words),
            "hash": self.hash,
            "tokenizer_names": list(self.tokenizer_names),
            "source": self.source,
            "build_params": dict(self.build_params),
            "size": self.size,
        }
