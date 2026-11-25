"""The strategy that finds the anchors of an answer.

An anchor is a pair (line of the source text, line of the answer) that the validator takes for one and
the same line. The anchors increase in both lines; the words between two neighbouring anchors are
decided by a strategy of ``gaps``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class AnchorFinder(Protocol):
    """A strategy that finds the anchors of an answer."""

    def find(
        self,
        reference_signatures: Sequence[tuple[str, ...]],
        answer_signatures: Sequence[tuple[str, ...]],
        sources: frozenset[str],
    ) -> list[tuple[int, int]]:
        """Return the anchors: pairs (line of the source text, line of the answer), increasing in both.

        Args:
            reference_signatures: The lines of the source text, the canonical form of the reference.
            answer_signatures: The canonical form of the answer lines.
            sources: The left-hand words of the rules.

        Returns:
            The anchors; an empty list when the text or the answer has no lines.
        """
        ...
