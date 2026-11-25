"""The alignment of an answer with the reference by places, in steps - the answer validator.

A place is a word of the reference or a word of the answer that has no place in the reference
(``aligned_answer``). The alignment goes in steps. The lines get their canonical form: the
right-hand word of every rule is replaced by its left-hand word, so a right replacement and an
unreplaced word look the same, and a line rewritten with any such outcomes equals the line of the
source text. The anchors - pairs of lines taken for one and the same line - are found by a strategy
of ``anchors``: by default an answer line is recognized by its background words, and of all
recognized pairs the anchors are the longest chain going forward in both texts. Lines without task
words at the edges of the answer are wrapper lines (``wrapper_lines``). Between anchors lie gaps; a
strategy of ``gaps`` decides every gap: what to pair, what is deleted, what is inserted. The
accumulator turns the anchors and the gap decisions into the arrays of places (``place_builder``):
in a pair of lines of equal length the words stand by positions, as in the article, otherwise by the
alignment with the fewest edits. Last, the cover of the places is checked (``conservation_check``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np

from iec.answer_validator.alignment.aligned_answer import AlignedAnswer
from iec.answer_validator.alignment.anchors import DEFAULT_ANCHOR_FINDER, AnchorFinder
from iec.answer_validator.alignment.answer_text import normalize_output, split_lines
from iec.answer_validator.alignment.conservation_check import check_positions
from iec.answer_validator.alignment.gaps.gap_resolver import DEFAULT_GAP_RESOLVER, GapResolver
from iec.answer_validator.alignment.place_builder import PlaceBuilder
from iec.answer_validator.alignment.wrapper_lines import wrapper_lines
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


class AnswerAligner:
    """The validator: aligns an answer with its reference by places.

    Built once with its strategies (specification §2.4): the anchors and the decision of the gaps come
    through the constructor as interfaces, so another strategy is a new class, not a change of this
    one. The working step :meth:`align` can be called any number of times.
    """

    def __init__(self, anchor_finder: AnchorFinder, gap_resolver: GapResolver) -> None:
        """Remember the strategies.

        Args:
            anchor_finder: The strategy that finds the anchors of an answer.
            gap_resolver: The strategy that decides a gap between anchors.
        """
        self._anchor_finder = anchor_finder
        self._gap_resolver = gap_resolver

    def align(
        self, answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable
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
        anchors = self._anchor_finder.find(
            lines.source, lines.signatures, frozenset(rule_table.sources)
        )
        task_words = frozenset(source_text.words) | frozenset(rule_table.targets)
        wrapper = wrapper_lines(lines.answer, anchors, task_words)
        answer_start, answer_words = _core_starts(lines.answer, wrapper)
        reference_start, reference_words = _line_starts(lines.reference)
        builder = PlaceBuilder(
            lines.source, lines.answer, lines.signatures, reference_start, answer_start
        )
        pairings, episodes = _place_lines(builder, lines, anchors, wrapper, self._gap_resolver)
        kinds, reference_positions, answer_positions, origins = builder.arrays()
        check_positions(reference_positions, answer_positions, reference_words, answer_words)
        core = [line for index, line in enumerate(lines.answer) if index not in wrapper]
        return AlignedAnswer(
            kinds=kinds,
            reference_positions=reference_positions,
            answer_positions=answer_positions,
            groups=origins,
            core_words=tuple(chain.from_iterable(core)),
            core_line_sizes=np.fromiter(map(len, core), dtype=np.int64, count=len(core)),
            answer_words=answer_words,
            wrapper_lines=len(wrapper),
            episodes=episodes,
            pairings=tuple(pairings),
        )


#: The validator of the bench: the anchors by background words and the gaps decided by their size.
DEFAULT_ALIGNER = AnswerAligner(DEFAULT_ANCHOR_FINDER, DEFAULT_GAP_RESOLVER)


def align_answer(
    answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable
) -> AlignedAnswer:
    """Align the answer of the model with the reference by places, by the validator of the bench.

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
    return DEFAULT_ALIGNER.align(answer, reference, source_text, rule_table)


def _task_lines(
    answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable
) -> _TaskLines:
    """Return the lines of the task and of the answer and the canonical form of the answer.

    The canonical form of a line replaces the right-hand word of every rule by its left-hand word.
    """
    canonical = {rule.target: rule.source for rule in rule_table.rules}
    answer_lines = split_lines(normalize_output(answer))
    return _TaskLines(
        reference=_word_lines(reference.words, reference.words_per_line),
        source=_word_lines(source_text.words, source_text.words_per_line),
        answer=answer_lines,
        signatures=[tuple(map(canonical.get, line, line)) for line in answer_lines],
    )


def _word_lines(words: tuple[str, ...], words_per_line: int) -> tuple[tuple[str, ...], ...]:
    """Return the lines of a word stream without assembling the text and cutting it again.

    The same as ``split_lines(assemble_text(words, words_per_line))``: words carry no whitespace,
    and a stream without words is one empty line.
    """
    if not words:
        return ((),)
    return tuple(
        words[start : start + words_per_line] for start in range(0, len(words), words_per_line)
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
    bounds = [*anchors, (len(lines.reference), len(lines.answer))]
    last_reference, last_answer = -1, -1
    index = 0
    while index < len(bounds):
        reference_index, answer_index = bounds[index]
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
        if reference_index == len(lines.reference):
            break
        count = 1
        while index + count < len(anchors) and bounds[index + count] == (
            reference_index + count,
            answer_index + count,
        ):
            count += 1
        builder.add_anchor_run(reference_index, answer_index, count)
        last_reference, last_answer = reference_index + count - 1, answer_index + count - 1
        index += count
    return pairings, episodes
