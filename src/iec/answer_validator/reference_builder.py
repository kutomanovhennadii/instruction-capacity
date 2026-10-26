"""Builder of the reference ``Y* = apply(R_m, X)`` - the "ideal executor" of the task."""

from __future__ import annotations

from iec.contracts import Reference, RuleTable, SourceText
from iec.infrastructure.exceptions import ContractViolationError
from iec.infrastructure.logging import LogContext, get_logger

_MODULE_NAME = "reference_builder"

_logger = get_logger(LogContext(module_name=_MODULE_NAME))


def build_reference(rule_table: RuleTable, source_text: SourceText) -> Reference:
    """Build the single correct answer to the task.

    The mapping ``sigma`` is built from the table (``source -> target``) and applied to the
    stream element by element: a target is replaced, background passes as is. Thanks to the
    generation decisions (M2) the application is unambiguous - the groups ``sources`` and
    ``targets`` do not intersect, and replacement words are absent from the stream, so
    cascades and double replacements are impossible in principle, and one pass is enough.
    The function is pure and deterministic: the same inputs give the same reference to the
    byte.

    Args:
        rule_table: The replacement table of the task; the order of rules does not matter
            here, only the unambiguity of the mapping does.
        source_text: The word stream of the task with the ``replacement_positions`` markup.

    Returns:
        ``Reference``: a stream of reference words of the same length and the same line
        width as the source - the reference differs from the task only by replaced words.

    Raises:
        ContractViolationError: the task markup is inconsistent with the table
            (``REFERENCE_MARKUP_MISMATCH``) - a desynchronization of the stand generators,
            not a property of the model answer.
    """
    mapping = {rule.source: rule.target for rule in rule_table.rules}
    words = tuple(mapping.get(word, word) for word in source_text.words)

    _verify_markup(rule_table, source_text, words)
    _logger.info(
        f"reference built: {len(words)} words, "
        f"{len(source_text.replacement_positions)} replacements by {rule_table.m} rules"
    )
    return Reference(words=words, words_per_line=source_text.words_per_line)


def _verify_markup(
    rule_table: RuleTable,
    source_text: SourceText,
    reference_words: tuple[str, ...],
) -> None:
    """Check the task markup against the positions actually replaced.

    The markup must point exactly at the rule targets: each marked position exists and
    carries a word from ``sources``, and no word outside the markup is touched by a
    replacement. Both directions are checked - otherwise markup pointing at background
    would give vague diagnostics instead of a targeted one.
    """
    identity = sorted(rule.source for rule in rule_table.rules if rule.source == rule.target)
    if identity:
        raise ContractViolationError(
            f"rule table contains identity rules, replacement is unobservable: {identity[:5]}",
            error_code="REFERENCE_MARKUP_MISMATCH",
        )

    sources = frozenset(rule_table.sources)
    marked = set(source_text.replacement_positions)

    out_of_range = sorted(index for index in marked if not 0 <= index < len(source_text.words))
    if out_of_range:
        raise ContractViolationError(
            f"replacement_positions point outside the word stream of {len(source_text.words)} "
            f"words: {out_of_range[:5]}",
            error_code="REFERENCE_MARKUP_MISMATCH",
        )

    not_a_target = sorted(index for index in marked if source_text.words[index] not in sources)
    if not_a_target:
        raise ContractViolationError(
            f"replacement_positions mark words absent from the rule table at indices "
            f"{not_a_target[:5]}",
            error_code="REFERENCE_MARKUP_MISMATCH",
        )

    changed = {
        index
        for index, (before, after) in enumerate(
            zip(source_text.words, reference_words, strict=True)
        )
        if before != after
    }
    if changed != marked:
        raise ContractViolationError(
            f"replaced positions {sorted(changed - marked)[:5]} are not marked and marked "
            f"positions {sorted(marked - changed)[:5]} were not replaced",
            error_code="REFERENCE_MARKUP_MISMATCH",
        )
