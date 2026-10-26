"""The reference answer ``Y*``: the word stream of the source text with replacements applied."""

from __future__ import annotations

from dataclasses import dataclass

from iec.contracts.source_text import assemble_text


@dataclass(frozen=True, slots=True)
class Reference:
    """The only correct answer to the task, built by a program without the model.

    The representation is the same as in :class:`~iec.contracts.source_text.SourceText`:
    the tuple ``words`` in reading order (exactly ``v`` long) is primary, and the text form
    is assembled by the :attr:`text` property with the same rule (lines of
    ``words_per_line`` words, the last one may be shorter). The reference and the source
    text must look the same in everything except the replaced words - so the line width is
    carried over from the source: the protocol requires lines to be kept, and the
    validator synchronizes on them.
    """

    words: tuple[str, ...]
    words_per_line: int

    @property
    def text(self) -> str:
        """Assemble the text: words separated by spaces, lines of ``words_per_line`` words."""
        return assemble_text(self.words, self.words_per_line)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the reference word stream."""
        return {
            "words": list(self.words),
            "words_per_line": self.words_per_line,
        }
