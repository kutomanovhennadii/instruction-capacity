"""The source text of the task ``X``: a word stream with the markup of checked replacements."""

from __future__ import annotations

from dataclasses import dataclass


def assemble_text(words: tuple[str, ...], words_per_line: int) -> str:
    """Assemble a text from a word stream: words separated by spaces, lines of ``words_per_line``.

    The only rule of assembling the text form in the bench. It is used both by
    :class:`SourceText` and by :class:`~iec.contracts.reference.Reference`: the protocol
    requires the reference to look like the source in everything except the replaced
    words - the coincidence of the rule is ensured by construction, not by a copy of code.
    """
    lines = [
        " ".join(words[start : start + words_per_line])
        for start in range(0, len(words), words_per_line)
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SourceText:
    """The word stream of the task and facts about it.

    The primary representation is the tuple ``words`` in reading order; the text form is
    deterministically assembled from it by the :attr:`text` property (lines of
    ``words_per_line`` words, the last one may be shorter), a second representation is not
    stored. ``replacement_positions`` are the indices of stream words covered by the rules
    (the consumer is the validator). ``len_tokens`` is the actual number of tokens of the
    ready text, measured once by tokenizer ``tokenizer_name``; it is a fact for logs and
    for the context window check, not a fitting target.
    """

    words: tuple[str, ...]
    words_per_line: int
    replacement_positions: tuple[int, ...]
    len_tokens: int
    tokenizer_name: str
    seed_text: int

    @property
    def text(self) -> str:
        """Assemble the text: words separated by spaces, lines of ``words_per_line`` words."""
        return assemble_text(self.words, self.words_per_line)

    @property
    def replacement_count(self) -> int:
        """Return the number of checked replacements in the task (= m x v/D by construction)."""
        return len(self.replacement_positions)

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the stream with markup and measured facts."""
        return {
            "words": list(self.words),
            "words_per_line": self.words_per_line,
            "replacement_positions": list(self.replacement_positions),
            "len_tokens": self.len_tokens,
            "tokenizer_name": self.tokenizer_name,
            "seed_text": self.seed_text,
            "replacement_count": self.replacement_count,
        }
