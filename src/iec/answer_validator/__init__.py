"""The answer validator: the reference answer and the alignment of an answer with it by places."""

from iec.answer_validator.alignment import (
    GROUP_ANCHOR,
    KIND_DELETED,
    KIND_INSERTED,
    KIND_MATCHED,
    PLACE_DELETED,
    PLACE_INSERTED,
    PLACE_MATCHED,
    AlignedAnswer,
    Place,
    align_answer,
    normalize_output,
    split_lines,
)
from iec.answer_validator.reference_builder import build_reference, rebuild_reference

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
    "build_reference",
    "normalize_output",
    "rebuild_reference",
    "split_lines",
]
