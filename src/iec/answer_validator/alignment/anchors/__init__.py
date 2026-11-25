"""The anchors of an answer: pairs of lines taken for one and the same line, found by background words."""

from iec.answer_validator.alignment.anchors.anchor_finder import AnchorFinder
from iec.answer_validator.alignment.anchors.longest_chain import longest_chain
from iec.answer_validator.alignment.anchors.marker_anchor_finder import (
    DEFAULT_ANCHOR_FINDER,
    MarkerAnchorFinder,
)
from iec.answer_validator.alignment.anchors.text_line_index import (
    AnswerLines,
    TextLineIndex,
    build_text_line_index,
    encode_answer_lines,
)

__all__ = [
    "DEFAULT_ANCHOR_FINDER",
    "AnchorFinder",
    "AnswerLines",
    "MarkerAnchorFinder",
    "TextLineIndex",
    "build_text_line_index",
    "encode_answer_lines",
    "longest_chain",
]
