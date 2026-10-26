"""Run validator: comparing the model answer ``Y`` with the reference ``Y*`` trial by trial."""

from __future__ import annotations

import bisect

from iec.contracts import (
    OUTCOME_CORRECT,
    OUTCOME_MISSING,
    OUTCOME_OTHER_WORD,
    OUTCOME_UNREPLACED,
    OUTCOME_WRONG_RULE_TARGET,
    RawOutput,
    Reference,
    RuleStat,
    RuleTable,
    SourceText,
    TrialRecord,
    TrialResult,
)
from iec.infrastructure.exceptions import ValidationError
from iec.infrastructure.logging import LogContext, get_logger
from iec.infrastructure.telemetry import TelemetryContext, telemetry_step

_MODULE_NAME = "run_validator"


def _telemetry_context(
    self: RunValidator,
    raw_output: RawOutput,
    reference: Reference,
    source_text: SourceText,
    rule_table: RuleTable,
) -> TelemetryContext:
    """Extract the matrix point key for the telemetry event of validation."""
    return TelemetryContext(
        m=rule_table.m,
        len_tokens=source_text.len_tokens,
        seed_rules=rule_table.seed_rules,
        seed_text=source_text.seed_text,
    )


def normalize_output(text: str) -> str:
    """Strip trailing whitespace from the model answer - and nothing more.

    The only concession to the answer: a line break at the end of generation carries no
    information about replacements. Case is not folded (vocabulary words are lowercase,
    ``Stone`` is a trial error), wrappers are not stripped (in v0 that is a structural
    failure): every concession weakens the instrument.
    """
    return text.rstrip()


def split_lines(text: str) -> tuple[tuple[str, ...], ...]:
    """Cut the text into lines, and the lines into words by whitespace.

    Lines are the synchronization anchors of the protocol, so splitting into them comes
    first: an empty line is kept as a line without words, otherwise an inserted line break
    would vanish without a trace instead of shifting the anchors.
    """
    return tuple(tuple(line.split()) for line in text.split("\n"))


def is_protocol_line(line: tuple[str, ...]) -> bool:
    """Decide whether a line consists only of words of the protocol alphabet.

    The protocol alphabet is lowercase ASCII letters: vocabulary words are exactly such, and
    there is no punctuation in the task text at all. An empty line does not belong to the
    alphabet: the reference never has empty lines, while in an answer it is a typical
    wrapper separator.
    """
    if not line:
        return False
    return all(word.isascii() and word.isalpha() and word.islower() for word in line)


class WrapperStrip:
    """Result of stripping the edge wrapper: the answer core and the count of stripped lines."""

    __slots__ = ("core_lines", "leading", "stripped", "trailing")

    def __init__(
        self,
        *,
        core_lines: tuple[tuple[str, ...], ...],
        leading: int,
        trailing: int,
        stripped: bool,
    ) -> None:
        self.core_lines = core_lines
        self.leading = leading
        self.trailing = trailing
        self.stripped = stripped


def strip_wrapper(
    answer_lines: tuple[tuple[str, ...], ...],
    reference_lines: tuple[tuple[str, ...], ...],
    *,
    start_anchors: frozenset[str],
    end_anchors: frozenset[str],
    words_per_line: int,
) -> WrapperStrip:
    """Strip the framing remarks of the model from the edges of the answer and return the core.

    Live models precede and finish the answer with their own messages ("Here is the
    rewritten text", a code block fence, "Let me know if..."), although the instruction
    demands only the rewritten text. Without stripping, such a wrapper shifts the line
    anchors and zeroes the trials of a perfectly executed task: the instrument would measure
    the manners of the model, not its capacity. The form violation is not forgiven - it is
    recorded separately by the ``wrapper_*`` fields of the result.

    The candidate core is computed by two independent stages; the first whose candidate
    passed the guard is accepted. The middle of the answer is never touched - only the
    edges are stripped.

    1. **Protocol alphabet** - consecutive lines not belonging to the alphabet (empty ones
       or ones with capitals, digits, punctuation, backticks) are stripped from the edges.
       The stage does not depend on the correctness of replacements inside the core, but
       is blind to a wrapper made of lowercase words.
    2. **Anchors by edge words** - it is known what the core must begin and end with: the
       first word is the correct replacement ``Y*[0]`` or the lost ``X[0]``, the last is
       ``Y*[v-1]`` or ``X[v-1]``. The stage catches a lowercase wrapper but stays silent
       when the edge word of the core itself turned out to be a measurement error.

    Stage 2 strips a line only if its width is **not equal to the reference one**: a useful
    protocol line carries exactly ``words_per_line`` words, and a word stream appended by
    the model repeats this width, while a remark in natural language does not hold it.
    Without this condition the stage would eat an extra line of words - a loss of control
    over the text, which must remain a failure of the 3rd kind. Stage 1 needs no such
    condition: it strips only lines outside the protocol alphabet, and the task word stream
    always belongs to it.

    The guard common to both stages: a candidate is accepted only if it contains exactly
    the reference number of lines. Without it stage 1 would eat damaged useful lines (a
    capital letter in the first line makes it "foreign"), turning local damage into a
    structural failure. If no stage passed - full rollback: the answer is validated whole
    by the former scheme, and the wrapper counters stay zero, since telling a wrapper from
    damage to the core is impossible here, and the instrument is not supposed to guess.

    Args:
        answer_lines: Lines of the normalized model answer.
        reference_lines: Lines of the reference - the source of the required number of core
            lines.
        start_anchors: Words the core may begin with (``Y*[0]`` and ``X[0]``).
        end_anchors: Words the core may end with (``Y*[v-1]`` and ``X[v-1]``).
        words_per_line: Protocol line width - by it stage 2 tells a model remark from an
            appended word stream.

    Returns:
        ``WrapperStrip``: the core and the number of stripped lines; on refusal - the
        original lines and zeros.
    """
    if len(answer_lines) <= len(reference_lines):
        # A wrapper adds lines; an answer not longer than the reference has nothing to strip,
        # and shortening it is all the more forbidden - that is already a loss of useful
        # lines, the business of the line-by-line scheme.
        return WrapperStrip(core_lines=answer_lines, leading=0, trailing=0, stripped=False)

    required = len(reference_lines)
    for bounds in (
        _alphabet_bounds(answer_lines),
        _anchor_bounds(
            answer_lines,
            start_anchors=start_anchors,
            end_anchors=end_anchors,
            words_per_line=words_per_line,
        ),
    ):
        if bounds is None:
            continue
        start, end = bounds
        if end - start != required:
            continue
        return WrapperStrip(
            core_lines=answer_lines[start:end],
            leading=start,
            trailing=len(answer_lines) - end,
            stripped=start > 0 or end < len(answer_lines),
        )
    return WrapperStrip(core_lines=answer_lines, leading=0, trailing=0, stripped=False)


def _alphabet_bounds(answer_lines: tuple[tuple[str, ...], ...]) -> tuple[int, int]:
    """Return the core bounds by the protocol alphabet: edges without "foreign" lines."""
    start = 0
    while start < len(answer_lines) and not is_protocol_line(answer_lines[start]):
        start += 1
    end = len(answer_lines)
    while end > start and not is_protocol_line(answer_lines[end - 1]):
        end -= 1
    return start, end


def _anchor_bounds(
    answer_lines: tuple[tuple[str, ...], ...],
    *,
    start_anchors: frozenset[str],
    end_anchors: frozenset[str],
    words_per_line: int,
) -> tuple[int, int] | None:
    """Return the core bounds by the reference edge words, or ``None`` if no anchor is found.

    The start is the first line whose first word belongs to the start anchors; the end is
    the last line whose last word belongs to the end anchors. Exactly the edge words of a
    line are checked: an anchor word in the middle of a wrapper line means nothing.

    Only lines whose width is **not** equal to the reference one are stripped: a line of
    full width is a continuation of the word stream, not a model remark, and its
    disappearance would hide a loss of control over the text.
    """
    start = next(
        (
            index
            for index, line in enumerate(answer_lines)
            if line and line[0] in start_anchors
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index + 1
            for index in range(len(answer_lines) - 1, start - 1, -1)
            if answer_lines[index] and answer_lines[index][-1] in end_anchors
        ),
        None,
    )
    if end is None:
        return None
    stripped_lines = list(answer_lines[:start]) + list(answer_lines[end:])
    if any(len(line) == words_per_line for line in stripped_lines):
        return None
    return start, end


class RunValidator:
    """Meter of one run: turns the raw model answer into trials ``z_1 ... z_H``.

    A two-phase module: initialization is light (the validator holds no run state), the
    working method :meth:`validate` is the whole measuring cycle.

    The boundary of responsibility is hard: **a bad model answer is data, not a stand
    error**. An empty answer, a fragment, garbage - a valid input and a valid measurement
    result (zeros in ``z``). An exception is raised only where the validator cannot produce
    a result by contract - on inconsistent inputs of the stand itself. Infrastructure
    failures (timeout, API break) do not get here at all: the runner separates them (M4).

    Matching is two-level, with line-by-line isolation of failures: answer lines are matched
    to reference lines by number, and the next line is synchronized anew even if the
    previous one is damaged. The scheme is conservative (the model is never overestimated),
    but a local failure costs the tail of one line, not the tail of the whole text.

    Inside a line, reachability of positions is decided by its length, not by word
    coincidence: a replaced word is a measured event, not a loss of synchronization, so a
    line of equal length is scored positionally as a whole. The opposite decision (cutting
    the line at the first mismatch) would attribute a structural failure to every ordinary
    replacement error.

    The failure kinds (a severity ladder, definitions in
    :class:`~iec.contracts.trial_result.TrialResult`) differ by absorption of the damage:
    absorbed by its own line - the 2nd kind, not absorbed (the next line is damaged too or
    the number of lines did not match) - the 3rd. Wrong replacements with an intact
    structure are the 1st kind, it lives in the trials themselves.
    """

    def __init__(self) -> None:
        """Prepare the validator; it has no dependencies and no run state."""
        self._logger = get_logger(LogContext(module_name=_MODULE_NAME))

    @telemetry_step("run_validate", module_name=_MODULE_NAME, context_extractor=_telemetry_context)
    def validate(
        self,
        raw_output: RawOutput,
        reference: Reference,
        source_text: SourceText,
        rule_table: RuleTable,
    ) -> TrialResult:
        """Score the model answer on each checked replacement.

        Args:
            raw_output: The raw model answer ``Y``; any of its text is a valid input.
            reference: The reference ``Y*`` built by ``build_reference``.
            source_text: The word stream of the task with the ``replacement_positions``
                markup.
            rule_table: The replacement table - the source of the expected words, of the rule
                positions and of telling "mixed up the rules" from "put a foreign word".

        Returns:
            ``TrialResult``: exactly ``H`` trial records in reading order, statistics per
            rule, the ladder of three failure kinds, the background counter and the strict
            metric.

        Raises:
            ValidationError: the stand inputs are inconsistent with each other
                (``VALIDATOR_INPUT_MISMATCH``) - the lengths of the reference and task
                streams differ, the markup refers beyond the stream, a marked word is not
                from the table. Properties of the model answer are never an exception.
        """
        self._check_inputs(reference, source_text, rule_table)

        answer = normalize_output(raw_output.text)
        p_strict = answer == reference.text

        answer_lines = split_lines(answer)
        reference_lines = split_lines(reference.text)
        wrapper = strip_wrapper(
            answer_lines,
            reference_lines,
            start_anchors=_edge_anchors(reference, source_text, position=0),
            end_anchors=_edge_anchors(reference, source_text, position=-1),
            words_per_line=reference.words_per_line,
        )
        if wrapper.stripped:
            self._logger.warning(
                f"answer is wrapped: {wrapper.leading} line(s) before and "
                f"{wrapper.trailing} after the rewritten text; trials are scored on the core"
            )
        observed, damage = self._align_lines(wrapper.core_lines, reference_lines)

        records = self._score_trials(observed, reference, source_text, rule_table)
        background_errors = self._count_background_errors(observed, reference, source_text)
        result = self._summarize(
            records, damage, background_errors, p_strict, rule_table, wrapper
        )

        self._logger.info(
            f"validated {result.h} trials: correct={result.correct_count} "
            f"p_event={result.p_event:.4f} p_strict={p_strict} "
            f"first_error_t={result.first_error_t} "
            f"fail_kind_2_line={result.first_fail_kind_2_line} "
            f"fail_kind_3_line={result.first_fail_kind_3_line} "
            f"broken_lines={result.broken_line_count} background_errors={background_errors} "
            f"wrapper_stripped={result.wrapper_stripped}"
        )
        if not p_strict:
            self._logger.warning(
                f"answer differs from the reference: err_rate={result.err_rate:.4f}, "
                f"{result.broken_line_count} lines with broken structure"
            )
        return result

    def _check_inputs(
        self,
        reference: Reference,
        source_text: SourceText,
        rule_table: RuleTable,
    ) -> None:
        """Make sure the reference, the task and the table describe one and the same task."""
        if len(reference.words) != len(source_text.words):
            raise ValidationError(
                f"reference stream of {len(reference.words)} words does not match the source "
                f"stream of {len(source_text.words)} words",
                error_code="VALIDATOR_INPUT_MISMATCH",
            )
        if reference.words_per_line != source_text.words_per_line:
            raise ValidationError(
                f"reference line width {reference.words_per_line} does not match the source "
                f"line width {source_text.words_per_line}",
                error_code="VALIDATOR_INPUT_MISMATCH",
            )

        stream_size = len(source_text.words)
        out_of_range = sorted(
            index for index in source_text.replacement_positions if not 0 <= index < stream_size
        )
        if out_of_range:
            raise ValidationError(
                f"replacement_positions point outside the word stream of {stream_size} words: "
                f"{out_of_range[:5]}",
                error_code="VALIDATOR_INPUT_MISMATCH",
            )

        sources = frozenset(rule_table.sources)
        not_a_target = sorted(
            index
            for index in source_text.replacement_positions
            if source_text.words[index] not in sources
        )
        if not_a_target:
            raise ValidationError(
                f"replacement_positions mark words absent from the rule table at indices "
                f"{not_a_target[:5]}",
                error_code="VALIDATOR_INPUT_MISMATCH",
            )

    def _align_lines(
        self,
        answer_lines: tuple[tuple[str, ...], ...],
        reference_lines: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str | None, ...], _LineDamage]:
        """Match lines by number and return the observed stream of length ``v``.

        A stream of the same length as the reference one is returned: in each position - the
        answer word if the position is reachable, otherwise ``None``.

        Reachability is decided **by line length, not by word coincidence**: replacing a word
        with another is a measured event (a trial error or background damage), the positions
        stay reliable, and zeroing the tail of the line because of it would attribute a
        structural failure to the model where there is none. Synchronization is broken only
        by a shift: if the length of the answer line did not match the reference one, the
        positions further inside the line mean nothing - the matched prefix is kept, and the
        tail is declared unreachable; the next line is synchronized anew by its anchor.

        The failure kind is determined by **absorption**: the damage is absorbed by its own
        line if the next line is recognizable (its word count matched the reference) or the
        damaged line is the last one. Absorbed - a failure of the 2nd kind, local. Not
        absorbed - a failure of the 3rd kind: control over the lines is lost, and spilling of
        words into the neighbouring line falls here as well, although the number of lines
        matches with it.

        A mismatch of the total number of lines is a failure of the 3rd kind by itself, in
        both directions: when all damages one by one look absorbed, but the answer has more
        or fewer lines than the reference, the failure is recorded on the first line beyond
        the end of the shorter side (the first extra one or the first missing one).

        The line of the failure of the 3rd kind is the place where the answer loses line
        synchronization with its text (``_sync_loss_line``): answer lines are tied to the
        reference by exact copies forward through the text, and the place is the first
        reference line after the last tie at its own number, if a tie with a shifted number
        follows. Otherwise, and never later, it is the line found above - the first
        unabsorbed damaged line or the first line beyond the end of the shorter side.
        """
        observed: list[str | None] = []
        broken_lines: list[int] = []

        for number, expected_line in enumerate(reference_lines, start=1):
            answer_line = answer_lines[number - 1] if number <= len(answer_lines) else None
            if answer_line is None:
                observed.extend([None] * len(expected_line))
                broken_lines.append(number)
                continue

            if len(answer_line) == len(expected_line):
                observed.extend(answer_line)
                continue

            intact = _intact_prefix(answer_line, expected_line)
            observed.extend(answer_line[:intact])
            observed.extend([None] * (len(expected_line) - intact))
            broken_lines.append(number)

        broken = frozenset(broken_lines)
        absorbed = [
            number for number in broken_lines if _is_absorbed(number, broken, reference_lines)
        ]
        unabsorbed = [number for number in broken_lines if number not in frozenset(absorbed)]

        # A mismatch of the number of lines is a loss of control over the lines by itself,
        # even when each damage one by one looks absorbed: an extra line in the tail damages
        # no reference line, a merge in the middle is masked by a shifted neighbour
        # recognizable by its word count, and the absence of the last line by its position.
        # The first line beyond the end of the shorter side is where the loss of control
        # became an observable fact by number; it is only the upper bound of the place, and
        # the place itself - the line of the drop or the merge - is found by
        # ``_sync_loss_line``.
        if not unabsorbed and len(answer_lines) != len(reference_lines):
            first_beyond = min(len(answer_lines), len(reference_lines)) + 1
            unabsorbed = [first_beyond]
            absorbed = [number for number in absorbed if number != first_beyond]

        return tuple(observed), _LineDamage(
            first_fail_kind_2_line=absorbed[0] if absorbed else None,
            first_fail_kind_3_line=(
                _sync_loss_line(answer_lines, reference_lines, unabsorbed[0])
                if unabsorbed
                else None
            ),
            broken_line_count=len(broken_lines),
        )

    def _score_trials(
        self,
        observed: tuple[str | None, ...],
        reference: Reference,
        source_text: SourceText,
        rule_table: RuleTable,
    ) -> tuple[TrialRecord, ...]:
        """Give a verdict and a category for each marked position in reading order."""
        rule_index_by_source = {
            rule.source: index for index, rule in enumerate(rule_table.rules, 1)
        }
        targets = frozenset(rule_table.targets)

        records = []
        for t, position in enumerate(sorted(source_text.replacement_positions), start=1):
            expected = reference.words[position]
            actual = observed[position]
            records.append(
                TrialRecord(
                    t=t,
                    rule_index=rule_index_by_source[source_text.words[position]],
                    word_position=position,
                    expected=expected,
                    observed=actual,
                    z=1 if actual == expected else 0,
                    outcome=_classify(
                        actual,
                        expected=expected,
                        source_word=source_text.words[position],
                        targets=targets,
                    ),
                )
            )
        return tuple(records)

    def _count_background_errors(
        self,
        observed: tuple[str | None, ...],
        reference: Reference,
        source_text: SourceText,
    ) -> int:
        """Count damaged background words in reachable positions.

        This is a violation of the ban "do not change the rest" - neither a trial nor a
        structural failure: unreachable positions are not counted here, they are already
        accounted for by the ``missing`` category of their trials and by the counter of
        lines with broken structure.
        """
        marked = frozenset(source_text.replacement_positions)
        return sum(
            1
            for position, word in enumerate(observed)
            if position not in marked and word is not None and word != reference.words[position]
        )

    def _summarize(
        self,
        records: tuple[TrialRecord, ...],
        damage: _LineDamage,
        background_errors: int,
        p_strict: bool,
        rule_table: RuleTable,
        wrapper: WrapperStrip,
    ) -> TrialResult:
        """Assemble the run summary from the trial records and the structural facts."""
        first_error = next((record for record in records if record.z == 0), None)
        return TrialResult(
            records=records,
            rule_stats=_rule_stats(records, rule_table),
            first_error_t=first_error.t if first_error is not None else None,
            first_error_rule_index=first_error.rule_index if first_error is not None else None,
            first_fail_kind_2_line=damage.first_fail_kind_2_line,
            first_fail_kind_3_line=damage.first_fail_kind_3_line,
            broken_line_count=damage.broken_line_count,
            background_errors=background_errors,
            p_strict=p_strict,
            wrapper_leading_lines=wrapper.leading,
            wrapper_trailing_lines=wrapper.trailing,
            wrapper_stripped=wrapper.stripped,
        )


class _LineDamage:
    """Structural facts of line-by-line matching: the failure ladder and its scale."""

    __slots__ = ("broken_line_count", "first_fail_kind_2_line", "first_fail_kind_3_line")

    def __init__(
        self,
        *,
        first_fail_kind_2_line: int | None,
        first_fail_kind_3_line: int | None,
        broken_line_count: int,
    ) -> None:
        self.first_fail_kind_2_line = first_fail_kind_2_line
        self.first_fail_kind_3_line = first_fail_kind_3_line
        self.broken_line_count = broken_line_count


def _edge_anchors(
    reference: Reference,
    source_text: SourceText,
    *,
    position: int,
) -> frozenset[str]:
    """Collect the words the answer core may begin or end with.

    The edge word of the core is either a correctly applied replacement (the reference word)
    or a lost one (the task word); for a background word both forms coincide. A pair, not
    one word: the most frequent error is a non-replacement, and the anchor must keep working
    on it.
    """
    return frozenset({reference.words[position], source_text.words[position]})


def _is_absorbed(
    number: int,
    broken: frozenset[int],
    reference_lines: tuple[tuple[str, ...], ...],
) -> bool:
    """Decide whether the damage of line ``number`` is absorbed by its own boundaries.

    Absorbed - if the next line is recognizable, that is not damaged: then the anchor after
    the failure was found and synchronization was restored. Damage of the last line is
    absorbed by definition - there is no need to restore synchronization any more.

    Recognizability is checked by the word count (that is what "damaged" means), not by
    content coincidence: otherwise an ordinary replacement error in the next line - a
    failure of the 1st kind - would falsely raise a local failure of the 2nd kind to the
    3rd.
    """
    if number == len(reference_lines):
        return True
    return number + 1 not in broken


def _sync_loss_line(
    answer_lines: tuple[tuple[str, ...], ...],
    reference_lines: tuple[tuple[str, ...], ...],
    kind_3_line: int,
) -> int:
    """Return the reference line (from 1) where the answer loses line synchronization.

    Answer lines are tied to reference lines only by exact copies and only forward through
    the text. While a tied answer line stands at its own number, matching by number is right;
    the first tie with a shifted number means lines were dropped or merged before it, and the
    synchronization was lost on the first reference line after the last tie at its own number.
    Without a shifted tie - a cut answer, a break inside a line, extra lines at the end - the
    place is ``kind_3_line``. The place is never later than ``kind_3_line``.
    """
    positions: dict[tuple[str, ...], list[int]] = {}
    for index, line in enumerate(reference_lines):
        positions.setdefault(line, []).append(index)
    last_in_sync = -1
    next_reference = 0
    for index, line in enumerate(answer_lines):
        hits = positions.get(line)
        if not hits:
            continue
        slot = bisect.bisect_left(hits, next_reference)
        if slot == len(hits):
            continue
        tied = hits[slot]
        next_reference = tied + 1
        if tied == index:
            last_in_sync = tied
        else:
            return min(last_in_sync + 2, kind_3_line)
    return kind_3_line


def _intact_prefix(answer_line: tuple[str, ...], expected_line: tuple[str, ...]) -> int:
    """Return the number of line words matched positionally before the first mismatch."""
    intact = 0
    for actual, expected in zip(answer_line, expected_line, strict=False):
        if actual != expected:
            break
        intact += 1
    return intact


def _classify(
    actual: str | None,
    *,
    expected: str,
    source_word: str,
    targets: frozenset[str],
) -> str:
    """Determine the trial category by what stands in its position.

    The categories distinguish degradation mechanisms that cannot be recovered later:
    ``unreplaced`` - the rule is lost, ``wrong_rule_target`` - the rules are mixed up with
    each other. The category does not enter the verdict ``z``, which is a strict 0/1.
    """
    if actual is None:
        return OUTCOME_MISSING
    if actual == expected:
        return OUTCOME_CORRECT
    if actual == source_word:
        return OUTCOME_UNREPLACED
    if actual in targets:
        return OUTCOME_WRONG_RULE_TARGET
    return OUTCOME_OTHER_WORD


def _rule_stats(records: tuple[TrialRecord, ...], rule_table: RuleTable) -> tuple[RuleStat, ...]:
    """Fold the records into statistics per rule in the order of the replacement table."""
    attempts = dict.fromkeys(range(1, rule_table.m + 1), 0)
    correct = dict.fromkeys(range(1, rule_table.m + 1), 0)
    for record in records:
        attempts[record.rule_index] += 1
        correct[record.rule_index] += record.z
    return tuple(
        RuleStat(rule_index=index, attempts=attempts[index], correct=correct[index])
        for index in range(1, rule_table.m + 1)
    )
