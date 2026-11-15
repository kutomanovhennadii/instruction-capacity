"""The alignment of an answer with the reference by places: the steps, the places and the gaps."""

from iec.answer_validator.alignment.aligned_answer import (
    GROUP_ANCHOR,
    KIND_DELETED,
    KIND_INSERTED,
    KIND_MATCHED,
    PLACE_DELETED,
    PLACE_INSERTED,
    PLACE_MATCHED,
    AlignedAnswer,
    Place,
)
from iec.answer_validator.alignment.answer_text import normalize_output, split_lines
from iec.answer_validator.alignment.line_alignment import align_answer

__all__ = [
    "GROUP_ANCHOR",
    "KIND_DELETED",
    "KIND_INSERTED",
    "KIND_MATCHED",
    "PLACE_DELETED",
    "PLACE_INSERTED",
    "PLACE_MATCHED",
    "AlignedAnswer",
    "Place",
    "align_answer",
    "normalize_output",
    "split_lines",
]
