"""Mirror tests of the run validator.

Each detector and each category is checked on a handmade mini task of two lines of three
words, where the expected result is written out by hand: the reference
``cat rock dog`` / ``lake cat rock`` for the source ``tree sun dog`` / ``lake tree sun`` and
four trials at positions 0, 1, 4, 5.
"""

import dataclasses

import pytest

from iec.answer_validator import RunValidator, build_reference
from iec.contracts import RawOutput, Reference, Rule, RuleTable, SourceText, TrialResult
from iec.infrastructure.exceptions import ValidationError

_TASK_WORDS = ("tree", "sun", "dog", "lake", "tree", "sun")
_PERFECT_ANSWER = "cat rock dog\nlake cat rock"


@pytest.fixture
def rule_table() -> RuleTable:
    """Return a table of two rules: ``tree -> cat`` (rule 1), ``sun -> rock`` (rule 2)."""
    return RuleTable(
        rules=(Rule(source="tree", target="cat"), Rule(source="sun", target="rock")),
        vocab_id="v_test",
        seed_rules=7,
    )


@pytest.fixture
def source_text() -> SourceText:
    """Return a handmade stream: two lines of three words, four targets in the markup."""
    return SourceText(
        words=_TASK_WORDS,
        words_per_line=3,
        replacement_positions=(0, 1, 4, 5),
        len_tokens=len(_TASK_WORDS),
        tokenizer_name="mock",
        seed_text=42,
    )


@pytest.fixture
def reference(rule_table: RuleTable, source_text: SourceText) -> Reference:
    """Return the reference of the handmade task, built by the production builder."""
    return build_reference(rule_table, source_text)


@pytest.fixture
def validate(
    rule_table: RuleTable,
    source_text: SourceText,
    reference: Reference,
):
    """Return the function "answer text -> TrialResult" for the handmade task."""

    def run(answer: str) -> TrialResult:
        return RunValidator().validate(RawOutput(text=answer), reference, source_text, rule_table)

    return run


def test_reference_text_is_the_perfect_answer(reference: Reference) -> None:
    """Readability insurance of the other tests: the reference is exactly the handwritten one."""
    assert reference.text == _PERFECT_ANSWER


def test_perfect_answer_gets_no_errors_at_all(validate) -> None:
    result = validate(_PERFECT_ANSWER)

    assert [record.z for record in result.records] == [1, 1, 1, 1]
    assert [record.outcome for record in result.records] == ["correct"] * 4
    assert result.p_event == 1.0
    assert result.err_rate == 0.0
    assert result.p_strict is True
    assert result.first_error_t is None
    assert result.first_error_rule_index is None
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0


def test_trailing_whitespace_is_the_only_concession(validate) -> None:
    result = validate(f"{_PERFECT_ANSWER}\n  \n")

    assert result.p_strict is True
    assert result.p_event == 1.0


def test_changed_case_is_an_error_not_a_normalization(validate) -> None:
    result = validate("Cat rock dog\nlake cat rock")

    assert result.records[0].z == 0
    assert result.records[0].outcome == "other_word"
    assert result.p_strict is False


def test_records_carry_the_fields_the_analysis_needs(validate) -> None:
    result = validate(_PERFECT_ANSWER)

    assert [(r.t, r.rule_index, r.word_position) for r in result.records] == [
        (1, 1, 0),
        (2, 2, 1),
        (3, 1, 4),
        (4, 2, 5),
    ]
    assert [(r.expected, r.observed) for r in result.records] == [
        ("cat", "cat"),
        ("rock", "rock"),
        ("cat", "cat"),
        ("rock", "rock"),
    ]


def test_unreplaced_target_is_a_lost_rule(validate) -> None:
    """The target stayed in place: the rule is lost, structure intact - failure of the 1st kind."""
    result = validate("tree rock dog\nlake cat rock")

    assert result.records[0].z == 0
    assert result.records[0].outcome == "unreplaced"
    assert result.records[0].observed == "tree"
    assert result.first_error_t == 1
    assert result.first_error_rule_index == 1
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0
    assert result.p_event == 0.75


def test_target_of_another_rule_is_confused_instructions(validate) -> None:
    """The target of rule 2 stands in the position of rule 1 - the model mixed up instructions."""
    result = validate("rock rock dog\nlake cat rock")

    assert result.records[0].outcome == "wrong_rule_target"
    assert result.records[0].z == 0
    assert result.first_error_rule_index == 1


def test_foreign_word_in_a_trial_position_is_other_word(validate) -> None:
    result = validate("cat rock dog\nlake wolf rock")

    assert result.records[2].outcome == "other_word"
    assert result.records[2].observed == "wolf"
    assert result.first_error_t == 3
    assert result.first_error_rule_index == 1


def test_corrupted_background_word_is_counted_apart_from_trials(validate) -> None:
    """Background damaged, structure intact: neither a trial nor a failure - a separate counter."""
    result = validate("cat rock wolf\nlake cat rock")

    assert result.background_errors == 1
    assert result.p_event == 1.0
    assert result.first_error_t is None
    assert result.broken_line_count == 0
    assert result.p_strict is False


def test_rule_stats_follow_the_table_order(validate) -> None:
    """Rule 1 lost both replacements, rule 2 executed both."""
    result = validate("tree rock dog\nlake tree rock")

    assert [(s.rule_index, s.attempts, s.correct, s.p_rule) for s in result.rule_stats] == [
        (1, 2, 0, 0.0),
        (2, 2, 2, 1.0),
    ]


def test_deleted_word_damages_only_its_own_line(validate) -> None:
    """Isolation of a failure by its line is a mandatory property of the scheme.

    A word is missing in the first line: from the point of divergence its tail is
    unreachable, but the second line is synchronized anew by its anchor and scored fully.
    """
    result = validate("cat dog\nlake cat rock")

    assert [record.outcome for record in result.records] == [
        "correct",
        "missing",
        "correct",
        "correct",
    ]
    assert [record.z for record in result.records] == [1, 0, 1, 1]
    assert result.records[1].observed is None
    assert result.first_fail_kind_2_line == 1
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 1
    assert result.background_errors == 0


def test_inserted_word_damages_only_its_own_line(validate) -> None:
    result = validate("cat wolf rock dog\nlake cat rock")

    assert [record.outcome for record in result.records] == [
        "correct",
        "missing",
        "correct",
        "correct",
    ]
    assert result.first_fail_kind_2_line == 1
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 1


def test_damage_in_the_second_line_leaves_the_first_intact(validate) -> None:
    result = validate("cat rock dog\nlake cat")

    assert [record.outcome for record in result.records] == [
        "correct",
        "correct",
        "correct",
        "missing",
    ]
    assert result.first_fail_kind_2_line == 2
    assert result.broken_line_count == 1


def test_glued_lines_are_a_fail_of_the_third_kind(validate) -> None:
    """Glued lines: the damage of line 1 is not absorbed - line 2 is absent from the answer.

    The glued line 1 carries six words instead of three; its matched prefix is kept, the
    tail is unreachable. Line 2 is absent, that is damaged too - so the anchor after the
    failure was not found: a failure of the 3rd kind on line 1. Line 2 itself is the last
    one, its damage is absorbed by definition, so the 2nd kind is recorded on it - the
    stages are independent.
    """
    result = validate("cat rock dog lake cat rock")

    assert result.first_fail_kind_3_line == 1
    assert result.first_fail_kind_2_line == 2
    assert result.broken_line_count == 2
    assert [record.outcome for record in result.records] == [
        "correct",
        "correct",
        "missing",
        "missing",
    ]


def test_extra_line_is_a_fail_of_the_third_kind(validate) -> None:
    """An extra line at the end: both reference lines are intact, but line control is lost.

    Appending a whole line damages no trial - all replacements are in place. The only trace
    left is a failure of the 3rd kind on the line beyond the reference. Without it the
    violation of the protocol "do not change the structure" would vanish from the result
    without a trace.
    """
    result = validate("cat rock dog\nlake cat rock\nwolf wolf wolf")

    assert result.first_fail_kind_3_line == 3
    assert result.first_fail_kind_2_line is None
    assert result.broken_line_count == 0
    assert result.p_event == 1.0
    assert result.p_strict is False


def test_missing_last_line_is_a_fail_of_the_third_kind(validate) -> None:
    """An answer without the last line: the number of lines did not match - control is lost.

    Absorption is silent here: the missing line is the last one and "absorbed by
    definition". But a missing line is a loss of control by itself - otherwise a truncated
    answer would pass as the 2nd kind.
    """
    result = validate("cat rock dog")

    assert result.first_fail_kind_3_line == 2
    assert result.first_fail_kind_2_line is None
    assert result.broken_line_count == 1
    assert [record.outcome for record in result.records] == [
        "correct",
        "correct",
        "missing",
        "missing",
    ]


def test_word_shifted_into_the_next_line_is_a_fail_of_the_third_kind(validate) -> None:
    """A word spilled into the neighbouring line is the 3rd kind, though the line count matched.

    ``dog`` moved from line 1 to line 2: both lines have a wrong word count, there is no
    absorption - no recognizable anchor after the failure was found, line control is lost.
    The sign "the number of lines matched" is useless here, absorption is what works.
    """
    result = validate("cat rock\ndog lake cat rock")

    assert result.first_fail_kind_3_line == 1
    assert result.first_fail_kind_2_line == 2
    assert result.broken_line_count == 2
    assert [record.outcome for record in result.records] == [
        "correct",
        "correct",
        "missing",
        "missing",
    ]


def test_replacement_error_in_the_next_line_does_not_escalate_the_second_kind(validate) -> None:
    """Recognizability is by word count, not by content.

    Line 1 lacks a word (the 2nd kind), and line 2 has a wrong replacement (the 1st kind).
    The word count of line 2 matched the reference, so the anchor holds and the failure must
    not be raised to the 3rd kind.
    """
    result = validate("cat dog\nlake wolf rock")

    assert result.first_fail_kind_2_line == 1
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 1
    assert result.records[2].outcome == "other_word"
    assert result.first_error_t == 2


def test_preamble_and_postscript_are_stripped_by_the_protocol_alphabet(validate) -> None:
    """The wrapper is stripped, replacements are counted, the form violation is recorded apart.

    Stage 1: the wrapper lines do not belong to the protocol alphabet (capitals, a colon, an
    exclamation mark), the useful lines do. Without stripping these two lines would shift
    the anchors and zero a perfectly executed task.
    """
    result = validate(f"Here is the rewritten text:\n{_PERFECT_ANSWER}\nLet me know!")

    assert result.p_event == 1.0
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 1
    assert result.wrapper_stripped is True
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0
    assert result.p_strict is False


def test_code_fence_and_blank_lines_are_stripped(validate) -> None:
    """A code block fence and blank separator lines are an edge wrapper too."""
    result = validate(f"```\n{_PERFECT_ANSWER}\n```")

    assert result.p_event == 1.0
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 1
    assert result.wrapper_stripped is True


def test_lowercase_wrapper_is_stripped_by_the_edge_anchors(validate) -> None:
    """Stage 2: a wrapper of lowercase words does not contradict the alphabet - anchors catch it.

    The core must begin with the word ``cat`` (the correct replacement of ``tree``) and end
    with the word ``rock``; the wrapper lines do not satisfy these edge words. The width of
    the wrapper lines differs from the reference one (3 words) - otherwise they are
    indistinguishable from the word stream.
    """
    result = validate(f"here is the rewritten text\n{_PERFECT_ANSWER}\nhope that helps you")

    assert result.p_event == 1.0
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 1
    assert result.wrapper_stripped is True


def test_edge_anchors_survive_an_unreplaced_first_word(validate) -> None:
    """The anchor is the pair {replacement, target}: a lost replacement in position 0 holds.

    The core begins with the word ``tree`` instead of ``cat`` - the most frequent model
    error, and extraction must survive it, recording the trial as ``unreplaced``.
    """
    result = validate("here is the rewritten text\ntree rock dog\nlake cat rock")

    assert result.wrapper_stripped is True
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 0
    assert result.records[0].outcome == "unreplaced"
    assert result.p_event == 0.75
    assert result.first_fail_kind_3_line is None


def test_a_lowercase_line_of_protocol_width_is_never_stripped(validate) -> None:
    """The boundary of stage 2: a line of reference width is a word stream, not a model remark.

    A line of task words appended by the model repeats the protocol width, and stripping it
    would hide a loss of control over the text. Stage 2 does not touch such lines - the
    violation honestly stays a failure of the 3rd kind, and the wrapper counters are zero.
    """
    result = validate(f"{_PERFECT_ANSWER}\nwolf wolf wolf")

    assert result.wrapper_stripped is False
    assert result.wrapper_trailing_lines == 0
    assert result.first_fail_kind_3_line == 3
    assert result.p_event == 1.0


def test_wrapper_and_damage_inside_the_core_do_not_interfere(validate) -> None:
    """Extraction and measurement are independent: the wrapper is stripped, the error found."""
    result = validate("Here is the rewritten text:\ncat rock dog\nlake wolf rock")

    assert result.wrapper_stripped is True
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 0
    assert result.records[2].outcome == "other_word"
    assert result.p_event == 0.75
    assert result.broken_line_count == 0


def test_lowercase_wrapper_with_a_broken_edge_word_is_not_stripped(validate) -> None:
    """The boundary of the scheme, recorded honestly.

    A lowercase wrapper is opaque to the alphabet, and a foreign word at position 0 breaks
    the start anchor - both stages are silent. The answer is scored by the former scheme:
    the extra lines honestly give a failure of the 3rd kind, the wrapper counters stay zero,
    since there is nothing to tell the wrapper from core damage here.
    """
    result = validate("here is the text\nwolf rock dog\nlake cat rock")

    assert result.wrapper_stripped is False
    assert result.wrapper_leading_lines == 0
    assert result.wrapper_trailing_lines == 0
    assert result.first_fail_kind_3_line is not None


def test_capitalized_first_word_without_a_wrapper_is_not_eaten(validate) -> None:
    """The guard on the number of lines protects a useful line from being eaten by stage 1.

    An answer without any wrapper, but the first word is damaged by case: the line is
    formally "foreign". Stripping would leave one line less and turn local damage into a
    structural failure - the candidate guard rejects it, and the error is caught by the
    trial, as it should be.
    """
    result = validate("Cat rock dog\nlake cat rock")

    assert result.wrapper_stripped is False
    assert result.wrapper_leading_lines == 0
    assert result.broken_line_count == 0
    assert result.first_fail_kind_3_line is None
    assert result.records[0].outcome == "other_word"
    assert result.p_event == 0.75


def test_perfect_answer_reports_no_wrapper(validate) -> None:
    """The instrument does not invent a wrapper where there is none."""
    result = validate(_PERFECT_ANSWER)

    assert result.wrapper_stripped is False
    assert result.wrapper_leading_lines == 0
    assert result.wrapper_trailing_lines == 0


def test_second_and_third_kind_coexist_in_one_result(
    rule_table: RuleTable, reference: Reference
) -> None:
    """The stages are independent: a local failure of the 2nd kind and a 3rd kind further on.

    A task of four lines of three words. A word disappeared in line 1, but line 2 is
    recognizable - absorbed, the 2nd kind. Lines 3 and 4 are both damaged (a word spilled
    from 3 into 4) - not absorbed, the 3rd kind on line 3.
    """
    words = (
        "tree",
        "sun",
        "dog",
        "lake",
        "tree",
        "sun",
        "tree",
        "sun",
        "dog",
        "lake",
        "tree",
        "sun",
    )
    source_text = SourceText(
        words=words,
        words_per_line=3,
        replacement_positions=(0, 1, 4, 5, 6, 7, 10, 11),
        len_tokens=len(words),
        tokenizer_name="mock",
        seed_text=42,
    )
    four_line_reference = build_reference(rule_table, source_text)
    assert four_line_reference.text == "cat rock dog\nlake cat rock\ncat rock dog\nlake cat rock"

    answer = "cat dog\nlake cat rock\ncat rock\ndog lake cat rock"
    result = RunValidator().validate(
        RawOutput(text=answer), four_line_reference, source_text, rule_table
    )

    assert result.first_fail_kind_2_line == 1
    assert result.first_fail_kind_3_line == 3
    assert result.broken_line_count == 3


def test_empty_answer_is_data_not_an_exception(validate) -> None:
    """An empty answer is a measurement result: all trials unreachable, zero correct."""
    result = validate("")

    assert result.h == 4
    assert result.correct_count == 0
    assert result.p_event == 0.0
    assert result.err_rate == 1.0
    assert [record.outcome for record in result.records] == ["missing"] * 4
    assert result.first_error_t == 1
    assert result.first_fail_kind_3_line == 1
    assert result.first_fail_kind_2_line == 2
    assert result.broken_line_count == 2
    assert result.background_errors == 0
    assert result.p_strict is False


def test_garbage_answer_is_data_not_an_exception(validate) -> None:
    result = validate("Here is your answer:\n```\nnothing useful\n```")

    assert result.p_event == 0.0
    assert result.p_strict is False


def test_reference_length_mismatch_is_an_input_error(
    rule_table: RuleTable, source_text: SourceText, reference: Reference
) -> None:
    broken = dataclasses.replace(reference, words=reference.words[:-1])

    with pytest.raises(ValidationError) as error:
        RunValidator().validate(RawOutput(text=_PERFECT_ANSWER), broken, source_text, rule_table)

    assert error.value.error_code == "VALIDATOR_INPUT_MISMATCH"


def test_line_width_mismatch_is_an_input_error(
    rule_table: RuleTable, source_text: SourceText, reference: Reference
) -> None:
    broken = dataclasses.replace(reference, words_per_line=10)

    with pytest.raises(ValidationError) as error:
        RunValidator().validate(RawOutput(text=_PERFECT_ANSWER), broken, source_text, rule_table)

    assert error.value.error_code == "VALIDATOR_INPUT_MISMATCH"


def test_markup_outside_the_stream_is_an_input_error(
    rule_table: RuleTable, source_text: SourceText, reference: Reference
) -> None:
    broken = dataclasses.replace(source_text, replacement_positions=(0, 1, 4, 99))

    with pytest.raises(ValidationError) as error:
        RunValidator().validate(RawOutput(text=_PERFECT_ANSWER), reference, broken, rule_table)

    assert error.value.error_code == "VALIDATOR_INPUT_MISMATCH"


def test_markup_of_a_background_word_is_an_input_error(
    rule_table: RuleTable, source_text: SourceText, reference: Reference
) -> None:
    broken = dataclasses.replace(source_text, replacement_positions=(0, 1, 2, 4, 5))

    with pytest.raises(ValidationError) as error:
        RunValidator().validate(RawOutput(text=_PERFECT_ANSWER), reference, broken, rule_table)

    assert error.value.error_code == "VALIDATOR_INPUT_MISMATCH"


def test_a_looping_answer_is_measured_as_a_kind_3_failure(task) -> None:
    """A looping answer is a measurement with a failure of the 3rd kind, not a discarded result.

    A live model can print an honest beginning, then repeat one line until it hits the
    ceiling, and the answer comes out **longer** than the reference. The validator parses such an answer by the
    ready rules: a mismatch of the number of lines is a loss of line control, unreachable
    positions go to ``missing`` of their trials. It got no edits for this case - the test
    fixes that it must not.
    """
    reference = build_reference(task.rule_table, task.source_text)
    lines = reference.text.split("\n")
    looped = "\n".join(lines[:2] + [lines[0]] * (len(lines) * 3))

    result = RunValidator().validate(
        RawOutput(text=looped), reference, task.source_text, task.rule_table
    )

    assert result.first_fail_kind_3_line is not None
    assert result.p_strict is False
    # The tail the model did not give verifies errors by zeros (axiom section 1 item 3).
    assert any(record.z == 0 for record in result.records)
    assert 0.0 <= result.p_event < 1.0


def test_dropped_lines_in_the_middle_are_placed_on_the_first_dropped_one(task) -> None:
    """Two lines dropped in the middle of a long text: the 3rd kind is placed where they fell out.

    The answer gives lines 1 and 2, skips lines 3 and 4 and rewrites the rest correctly. By
    number the loss shows only at the end, where the answer runs out of lines; by content the
    answer keeps step with its text up to line 2, and its next exact copy - reference line 5 -
    stands at number 3. The failure of the 3rd kind is recorded on line 3, the first dropped
    one.
    """
    reference = build_reference(task.rule_table, task.source_text)
    lines = reference.text.split("\n")
    assert len(lines) >= 5
    assert len(set(lines)) == len(lines)  # each exact copy points at one reference line
    answer = "\n".join(lines[:2] + lines[4:])

    result = RunValidator().validate(
        RawOutput(text=answer), reference, task.source_text, task.rule_table
    )

    assert result.first_fail_kind_3_line == 3


def test_a_cut_answer_is_placed_on_the_first_line_it_did_not_give(task) -> None:
    """An answer cut after line 3: the 3rd kind stays on line 4, as with matching by number.

    Every given line stands at its own number and there is no shifted exact copy, so the
    place of the failure is the first line beyond the end of the answer.
    """
    reference = build_reference(task.rule_table, task.source_text)
    lines = reference.text.split("\n")
    assert len(lines) >= 5
    answer = "\n".join(lines[:3])

    result = RunValidator().validate(
        RawOutput(text=answer), reference, task.source_text, task.rule_table
    )

    assert result.first_fail_kind_3_line == 4


def test_an_extra_word_in_a_middle_line_is_not_a_fail_of_the_third_kind(task) -> None:
    """One line in the middle carries an extra word, the rest is correct - the 2nd kind only.

    The damage is absorbed by the next line, recognizable by its word count, and the number
    of lines matches: there is no failure of the 3rd kind, and looking for its place by
    content does not invent one.
    """
    reference = build_reference(task.rule_table, task.source_text)
    lines = reference.text.split("\n")
    assert len(lines) >= 5
    lines[2] = f"wolf {lines[2]}"
    answer = "\n".join(lines)

    result = RunValidator().validate(
        RawOutput(text=answer), reference, task.source_text, task.rule_table
    )

    assert result.first_fail_kind_2_line == 3
    assert result.first_fail_kind_3_line is None


def test_generated_task_validates_its_own_reference_without_false_alarms(task) -> None:
    """A reference answer validates without false alarms (the full FP/FN check is in test_fp_fn)."""
    reference = build_reference(task.rule_table, task.source_text)

    result = RunValidator().validate(
        RawOutput(text=reference.text), reference, task.source_text, task.rule_table
    )

    assert result.h == task.source_text.replacement_count
    assert result.p_event == 1.0
    assert result.p_strict is True
    assert all(stat.p_rule == 1.0 for stat in result.rule_stats)
    assert result.broken_line_count == 0
    assert result.background_errors == 0
