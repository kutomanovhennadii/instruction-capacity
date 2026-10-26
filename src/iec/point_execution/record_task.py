"""Rebuilding the task of a point from its published record."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from iec.contracts import PointRecord, Rule, RuleTable, SourceText
from iec.model_client.tokenizer import Tokenizer
from iec.prompt_generator.text_generator import WORDS_PER_LINE

#: Tokenizer name of a source text whose length was not measured on reading.
UNMEASURED_TOKENIZER = "unmeasured"


def replacement_positions(words: Sequence[str], sources: Iterable[str]) -> tuple[int, ...]:
    """Return the indices of the words that the rules replace, in reading order.

    A position is replaced when its word stands on the left side of a rule. The words to
    replace and the replacements never intersect, and replacements never occur in the
    source text, so the markup is unambiguous.
    """
    source_set = frozenset(sources)
    return tuple(index for index, word in enumerate(words) if word in source_set)


def task_from_record(
    record: PointRecord, vocab_id: str, tokenizer: Tokenizer | None = None
) -> tuple[RuleTable, SourceText]:
    """Rebuild the rule table and the marked source text of a published point.

    Args:
        record: Published point record.
        vocab_id: Vocabulary of the measurement.
        tokenizer: Tokenizer to measure the text length; ``None`` - not measured, the
            length is ``0`` and the tokenizer name is ``"unmeasured"``.

    Returns:
        The rule table in the order shown to the model and the source text whose text
        form equals ``record.source_text``.
    """
    table = RuleTable(
        rules=tuple(Rule(source=source, target=target) for source, target in record.rules),
        vocab_id=vocab_id,
        seed_rules=record.conditions.seed_rules,
    )
    words = tuple(record.source_text.split())
    return table, SourceText(
        words=words,
        words_per_line=WORDS_PER_LINE,
        replacement_positions=replacement_positions(words, table.sources),
        len_tokens=0 if tokenizer is None else tokenizer.count_tokens(record.source_text),
        tokenizer_name=UNMEASURED_TOKENIZER if tokenizer is None else tokenizer.name,
        seed_text=record.conditions.seed_text,
    )
