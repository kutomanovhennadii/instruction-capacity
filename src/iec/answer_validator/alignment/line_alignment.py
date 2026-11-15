"""The alignment of an answer with the reference by places, in steps - the answer validator.

A place is a word of the reference or a word of the answer that has no place in the reference
(``aligned_answer``). The alignment goes in steps. The lines get their canonical form: the
right-hand word of every rule is replaced by its left-hand word, so a right replacement and an
unreplaced word look the same, and a line rewritten with any such outcomes equals the line of the
source text. Exactly equal lines are anchors, and lines without task words at the edges of the
answer are wrapper lines (``line_anchors``). Between anchors lie gaps; a strategy of ``gaps``
decides every gap: what to pair, what is deleted, what is inserted. The accumulator turns the
anchors and the gap decisions into the arrays of places (``place_builder``): in a pair of lines of
equal length the words stand by positions, as in the article, otherwise by the alignment with the
fewest edits. Last, the cover of the places is checked (``conservation_check``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from iec.answer_validator.alignment.aligned_answer import AlignedAnswer
from iec.answer_validator.alignment.answer_text import normalize_output, split_lines
from iec.answer_validator.alignment.conservation_check import check_positions
from iec.answer_validator.alignment.gaps.gap_resolver import DEFAULT_GAP_RESOLVER, GapResolver
from iec.answer_validator.alignment.line_anchors import anchor_lines, wrapper_lines
from iec.answer_validator.alignment.place_builder import PlaceBuilder
from iec.contracts import Reference, RuleTable, SourceText


@dataclass(frozen=True, slots=True)
class _TaskLines:
    """The lines of one task and one answer.

    ``reference`` - the reference lines; ``source`` - the source lines, the canonical form of the
    reference lines; ``answer`` - the answer lines; ``signatures`` - the canonical form of the
    answer lines.
    """

    reference: tuple[tuple[str, ...], ...]
    source: tuple[tuple[str, ...], ...]
    answer: tuple[tuple[str, ...], ...]
    signatures: list[tuple[str, ...]]


def align_answer(
    answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable
) -> AlignedAnswer:
    """Align the answer of the model with the reference by places.

    Args:
        answer: The raw answer of the model; any text is a valid input.
        reference: The reference built by ``build_reference``.
        source_text: The source text of the task.
        rule_table: The rule table in the order shown to the model.

    Returns:
        ``AlignedAnswer``.

    Raises:
        ContractViolationError: ``ALIGNMENT_CONSERVATION`` - the places do not cover the
            words of the reference or of the answer core exactly once; this is a defect of the
            code, not a property of the answer.
    """
    lines = _task_lines(answer, reference, source_text, rule_table)
    anchors = anchor_lines(lines.source, lines.signatures)
    task_words = frozenset(source_text.words) | frozenset(rule_table.targets)
    wrapper = wrapper_lines(lines.answer, anchors, task_words)
    answer_start, answer_words = _core_starts(lines.answer, wrapper)
    reference_start, reference_words = _line_starts(lines.reference)
    builder = PlaceBuilder(
        lines.source, lines.answer, lines.signatures, reference_start, answer_start
    )
    pairings, episodes = _place_lines(builder, lines, anchors, wrapper, DEFAULT_GAP_RESOLVER)
    kinds, reference_positions, answer_positions, origins = builder.arrays()
    check_positions(reference_positions, answer_positions, reference_words, answer_words)
    core = [line for index, line in enumerate(lines.answer) if index not in wrapper]
    return AlignedAnswer(
        kinds=kinds,
        reference_positions=reference_positions,
        answer_positions=answer_positions,
        groups=origins,
        core_words=tuple(word for line in core for word in line),
        core_line_sizes=np.fromiter(map(len, core), dtype=np.int64, count=len(core)),
        answer_words=answer_words,
        wrapper_lines=len(wrapper),
        episodes=episodes,
        pairings=tuple(pairings),
    )


def _task_lines(
    answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable
) -> _TaskLines:
    """Return the lines of the task and of the answer and the canonical form of the answer.

    The canonical form of a line replaces the right-hand word of every rule by its left-hand word.
    """
    canonical = {rule.target: rule.source for rule in rule_table.rules}
    answer_lines = split_lines(normalize_output(answer))
    return _TaskLines(
        reference=split_lines(reference.text),
        source=split_lines(source_text.text),
        answer=answer_lines,
        signatures=[tuple(map(canonical.get, line, line)) for line in answer_lines],
    )


def _core_starts(
    answer_lines: Sequence[tuple[str, ...]], wrapper: frozenset[int]
) -> tuple[dict[int, int], int]:
    """Return where every core line starts in the answer core and the number of core words."""
    answer_start: dict[int, int] = {}
    answer_words = 0
    for index, line in enumerate(answer_lines):
        if index not in wrapper:
            answer_start[index] = answer_words
            answer_words += len(line)
    return answer_start, answer_words


def _line_starts(reference_lines: Sequence[tuple[str, ...]]) -> tuple[list[int], int]:
    """Return where every reference line starts in the reference and the number of its words."""
    reference_start: list[int] = []
    reference_words = 0
    for line in reference_lines:
        reference_start.append(reference_words)
        reference_words += len(line)
    return reference_start, reference_words


def _place_lines(
    builder: PlaceBuilder,
    lines: _TaskLines,
    anchors: Sequence[tuple[int, int]],
    wrapper: frozenset[int],
    resolver: GapResolver,
) -> tuple[list[tuple[str, float]], int]:
    """Put the gaps and the anchors into places in the order of the lines.

    Every gap is decided by the strategy ``resolver``. An episode is a gap whose decision adds a
    deletion or an insertion: a synchronisation failure.

    Returns:
        The pairings of all gaps with the share of matched words of the shorter side, and the
        number of episodes.
    """
    copies = frozenset(lines.source)
    pairings: list[tuple[str, float]] = []
    episodes = 0
    last_reference, last_answer = -1, -1
    for reference_index, answer_index in (*anchors, (len(lines.reference), len(lines.answer))):
        gap_reference = list(range(last_reference + 1, reference_index))
        gap_answer = [j for j in range(last_answer + 1, answer_index) if j not in wrapper]
        if gap_reference or gap_answer:
            groups, gap_pairings = resolver.resolve(
                gap_reference, gap_answer, lines.source, lines.signatures, copies
            )
            pairings.extend(gap_pairings)
            unmatched = builder.unmatched
            for group in groups:
                builder.add_group(group)
            episodes += int(builder.unmatched > unmatched)
        if reference_index < len(lines.reference):
            builder.add_anchor(reference_index, answer_index)
        last_reference, last_answer = reference_index, answer_index
    return pairings, episodes
