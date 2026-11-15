"""The decision of the gaps between anchors: two strategies behind one interface."""

from iec.answer_validator.alignment.gaps.gap_decision import GapDecision, Group
from iec.answer_validator.alignment.gaps.gap_resolver import (
    DEFAULT_GAP_RESOLVER,
    GapResolver,
    SizedGapResolver,
)
from iec.answer_validator.alignment.gaps.large_gap import LargeGapResolver
from iec.answer_validator.alignment.gaps.small_gap import SmallGapResolver

__all__ = [
    "DEFAULT_GAP_RESOLVER",
    "GapDecision",
    "GapResolver",
    "Group",
    "LargeGapResolver",
    "SizedGapResolver",
    "SmallGapResolver",
]
