"""The conservation of places: every reference word and every word of the answer core once.

The check guards the instrument, not the answer: a bad answer gives deletions and insertions, but
the places always cover every word of the reference and of the answer core exactly once. A broken
cover is a defect of the code.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from iec.answer_validator.alignment.aligned_answer import Place
from iec.infrastructure.exceptions import ContractViolationError

#: Error code of places that do not cover the words exactly once.
ALIGNMENT_CONSERVATION = "ALIGNMENT_CONSERVATION"


def check_conservation(places: Sequence[Place], reference_words: int, answer_words: int) -> None:
    """Check that every reference word and every word of the answer core is in exactly one place.

    Raises:
        ContractViolationError: ``ALIGNMENT_CONSERVATION``.
    """
    check_positions(
        np.array(
            [-1 if p.reference_position is None else p.reference_position for p in places],
            dtype=np.int64,
        ),
        np.array(
            [-1 if p.answer_position is None else p.answer_position for p in places],
            dtype=np.int64,
        ),
        reference_words,
        answer_words,
    )


def check_positions(
    reference_positions: np.ndarray,
    answer_positions: np.ndarray,
    reference_words: int,
    answer_words: int,
) -> None:
    """Check by counting that the positions cover ``0..n-1`` exactly once on both sides.

    Args:
        reference_positions: The reference positions of the places, ``-1`` for none.
        answer_positions: The answer positions of the places, ``-1`` for none.
        reference_words: The number of words of the reference.
        answer_words: The number of words of the answer core.

    Raises:
        ContractViolationError: ``ALIGNMENT_CONSERVATION`` - a word is in no place or in two.
    """
    reference_seen = reference_positions[reference_positions >= 0]
    answer_seen = answer_positions[answer_positions >= 0]
    if not (
        np.array_equal(
            np.bincount(reference_seen, minlength=reference_words), np.ones(reference_words)
        )
        and np.array_equal(np.bincount(answer_seen, minlength=answer_words), np.ones(answer_words))
    ):
        raise ContractViolationError(
            f"alignment covers {len(reference_seen)} of {reference_words} reference words and "
            f"{len(answer_seen)} of {answer_words} answer words, not each exactly once",
            error_code=ALIGNMENT_CONSERVATION,
        )
