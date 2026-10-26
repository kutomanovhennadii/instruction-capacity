"""The validator criterion - "detector FP/FN are small" - and an end-to-end run from task to trials.

Two families of checks. "Ideal executor": the reference
submitted as the answer must not get a single attributed error - zero false alarms on the
parameter grid. Reference mutator: deterministic targeted damage of a correct answer, each of
which the validator must find exactly in the known place - exact ``t`` with ``z = 0``, exact
category, exact failure kind and its line - and nothing more.

The mutator lives right in this file: it is a checking tool, not production code, and
neighbouring helper modules are not importable under ``--import-mode=importlib`` without
packages.
"""

import json

import pytest

from iec.answer_validator import RunValidator, build_reference
from iec.contracts import RawOutput, Reference, TrialResult

# A word absent from the conftest test vocabulary: "foreign word" damage and background damage.
FOREIGN_WORD = "quasar"

# --- Reference mutator: each function introduces exactly one damage at a known address ---


def replace_word(reference: Reference, position: int, word: str) -> str:
    """Put ``word`` into stream position ``position`` - damage without a structure shift."""
    lines = _lines(reference)
    row, column = divmod(position, reference.words_per_line)
    lines[row][column] = word
    return _join(lines)


def delete_word(reference: Reference, position: int) -> str:
    """Delete the word at position ``position`` - its line becomes one word shorter."""
    lines = _lines(reference)
    row, column = divmod(position, reference.words_per_line)
    del lines[row][column]
    return _join(lines)


def insert_word(reference: Reference, line_number: int, word: str) -> str:
    """Insert ``word`` at the start of line ``line_number`` (1..) - the line is one word longer."""
    lines = _lines(reference)
    lines[line_number - 1].insert(0, word)
    return _join(lines)


def drag_word_to_next_line(reference: Reference, line_number: int) -> str:
    """Drag the last word of line ``line_number`` to the start of the next one.

    The number of lines does not change - the sign "the number of lines did not match" does
    not catch such a failure, absorption must tell it apart.
    """
    lines = _lines(reference)
    lines[line_number].insert(0, lines[line_number - 1].pop())
    return _join(lines)


def glue_lines(reference: Reference, line_number: int) -> str:
    """Glue line ``line_number`` with the next one - there is one line less."""
    lines = _lines(reference)
    lines[line_number - 1].extend(lines.pop(line_number))
    return _join(lines)


def _lines(reference: Reference) -> list[list[str]]:
    """Split the reference into mutable lines - by the same rule as the ``text`` assembly."""
    width = reference.words_per_line
    return [
        list(reference.words[start : start + width])
        for start in range(0, len(reference.words), width)
    ]


def _join(lines: list[list[str]]) -> str:
    """Assemble the answer text from lines: words by spaces, lines by line breaks."""
    return "\n".join(" ".join(line) for line in lines)


# --- Fixtures of the standard task (conftest: v=60, D=6, m=2 -> 20 trials, 6 lines) ---


@pytest.fixture
def reference(task) -> Reference:
    """Return the reference of the standard task, built by the production builder."""
    return build_reference(task.rule_table, task.source_text)


@pytest.fixture
def validate(task, reference):
    """Return the function "answer text -> TrialResult" for the standard task."""

    def run(answer: str) -> TrialResult:
        return RunValidator().validate(
            RawOutput(text=answer), reference, task.source_text, task.rule_table
        )

    return run


def _trial_positions(task) -> list[int]:
    """Return trial positions in reading order: element ``t - 1`` is trial ``t``."""
    return sorted(task.source_text.replacement_positions)


# --- Zero false alarms: the ideal executor on the parameter grid ---

_GRID = [
    # (D, n, m, seed_rules, seed_text); text length v = D*n
    (6, 10, 2, 7, 42),  # the standard point
    (4, 10, 1, 11, 137),  # one rule
    (6, 15, 3, 3, 5),
    (9, 5, 3, 17, 23),  # v is not a multiple of the line width - the last line is shorter
    (6, 5, 6, 1, 2),  # D = m: the lexicon consists entirely of targets, no background
    (8, 15, 4, 21, 99),
]


@pytest.mark.parametrize(("distinct", "appearances", "m", "seed_rules", "seed_text"), _GRID)
def test_ideal_executor_gets_zero_false_positives(
    make_task, distinct, appearances, m, seed_rules, seed_text
) -> None:
    """The validator must not attribute errors to an error-free answer - on the whole grid."""
    task = make_task(
        D=distinct,
        n=appearances,
        m=m,
        seed_rules=seed_rules,
        seed_text=seed_text,
    )
    reference = build_reference(task.rule_table, task.source_text)

    result = RunValidator().validate(
        RawOutput(text=reference.text), reference, task.source_text, task.rule_table
    )

    assert result.h == m * appearances
    assert all(record.z == 1 for record in result.records)
    assert all(record.outcome == "correct" for record in result.records)
    assert all(stat.attempts == appearances for stat in result.rule_stats)
    assert all(stat.p_rule == 1.0 for stat in result.rule_stats)
    assert result.p_event == 1.0
    assert result.p_strict is True
    assert result.first_error_t is None
    assert result.first_error_rule_index is None
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0


# --- Zero misses: targeted damage without a structure shift (failure of the 1st kind) ---


@pytest.mark.parametrize("t", [1, 7, 20])
def test_unreplaced_target_is_found_exactly(task, reference, validate, t) -> None:
    """A target left in trial ``t``: exactly that trial, category unreplaced, nothing more."""
    position = _trial_positions(task)[t - 1]
    answer = replace_word(reference, position, task.source_text.words[position])

    result = validate(answer)

    record = result.records[t - 1]
    assert record.z == 0
    assert record.outcome == "unreplaced"
    assert record.observed == task.source_text.words[position]
    assert [r.t for r in result.records if r.z == 0] == [t]
    assert result.first_error_t == t
    assert result.first_error_rule_index == record.rule_index
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0
    assert result.p_event == (result.h - 1) / result.h


def test_wrong_rule_target_is_found_exactly(task, reference, validate) -> None:
    """The target of another rule in a trial: the category tells "mixed up" from "lost"."""
    t = 3
    position = _trial_positions(task)[t - 1]
    own_index = task.rule_table.sources.index(task.source_text.words[position]) + 1
    alien_target = task.rule_table.targets[(1 if own_index == 2 else 2) - 1]
    answer = replace_word(reference, position, alien_target)

    result = validate(answer)

    record = result.records[t - 1]
    assert record.z == 0
    assert record.outcome == "wrong_rule_target"
    assert record.observed == alien_target
    assert record.rule_index == own_index
    assert [r.t for r in result.records if r.z == 0] == [t]
    assert result.first_error_t == t
    assert result.broken_line_count == 0
    assert result.background_errors == 0


def test_foreign_word_in_a_trial_is_found_exactly(task, vocab, reference, validate) -> None:
    """A foreign word in a trial: category other_word, nothing more."""
    t = 12
    position = _trial_positions(task)[t - 1]
    assert FOREIGN_WORD not in vocab.words
    answer = replace_word(reference, position, FOREIGN_WORD)

    result = validate(answer)

    record = result.records[t - 1]
    assert record.z == 0
    assert record.outcome == "other_word"
    assert record.observed == FOREIGN_WORD
    assert [r.t for r in result.records if r.z == 0] == [t]
    assert result.first_error_t == t
    assert result.broken_line_count == 0
    assert result.background_errors == 0


def test_corrupted_background_word_is_found_exactly(task, vocab, reference, validate) -> None:
    """Background damage: neither a trial nor a structural failure - background_errors only."""
    marked = frozenset(task.source_text.replacement_positions)
    position = next(index for index in range(len(reference.words)) if index not in marked)
    assert FOREIGN_WORD not in vocab.words
    answer = replace_word(reference, position, FOREIGN_WORD)

    result = validate(answer)

    assert result.background_errors == 1
    assert all(record.z == 1 for record in result.records)
    assert result.p_event == 1.0
    assert result.first_error_t is None
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.p_strict is False


# --- Zero misses: structure damage (failures of the 2nd and 3rd kind) ---


def test_deleted_word_zeroes_only_the_tail_of_its_line(task, reference, validate) -> None:
    """Mandatory isolation test: a failure of the 2nd kind costs the tail of one line, not text.

    A word is deleted at a trial position not in the last line: the trials of the tail of
    that line are unreachable, while the trials of all following lines are scored fully.
    """
    width = reference.words_per_line
    last_row = (len(reference.words) - 1) // width
    position = next(
        p
        for p in _trial_positions(task)
        if p // width != last_row
        and p % width != width - 1
        # The neighbouring word differs - the shift is visible right at the deletion point,
        # the damage address is exact.
        and reference.words[p] != reference.words[p + 1]
    )
    row, column = divmod(position, width)

    result = validate(delete_word(reference, position))

    tail = [
        r
        for r in result.records
        if r.word_position // width == row and r.word_position % width >= column
    ]
    others = [r for r in result.records if r not in tail]
    later_lines = [r for r in result.records if r.word_position // width > row]
    assert tail
    assert later_lines  # isolation has something to prove
    assert all(r.z == 0 and r.outcome == "missing" and r.observed is None for r in tail)
    assert all(r.z == 1 and r.outcome == "correct" for r in others)
    assert result.first_fail_kind_2_line == row + 1
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 1
    assert result.background_errors == 0


def test_inserted_word_zeroes_only_its_own_line(task, vocab, reference, validate) -> None:
    """An extra word at the start of a line: the whole line is unreachable, neighbours intact."""
    width = reference.words_per_line
    row = _trial_positions(task)[0] // width  # the line of the first trial
    assert row < (len(reference.words) - 1) // width
    assert FOREIGN_WORD not in vocab.words

    result = validate(insert_word(reference, row + 1, FOREIGN_WORD))

    in_line = [r for r in result.records if r.word_position // width == row]
    others = [r for r in result.records if r.word_position // width != row]
    assert in_line
    assert all(r.z == 0 and r.outcome == "missing" for r in in_line)
    assert all(r.z == 1 and r.outcome == "correct" for r in others)
    assert result.first_fail_kind_2_line == row + 1
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 1


def test_dragged_word_is_kind_3_despite_equal_line_count(task, reference, validate) -> None:
    """Mandatory spill test: the number of lines matched, but control is lost - the 3rd kind."""
    width = reference.words_per_line
    lines = reference.text.split("\n")
    # The dragged word differs from the first word of the next line - both lines diverge
    # from the anchors exactly where the damage is introduced.
    line_number = next(
        n for n in range(1, len(lines)) if lines[n - 1].split()[-1] != lines[n].split()[0]
    )
    answer = drag_word_to_next_line(reference, line_number)
    assert answer.count("\n") == reference.text.count("\n")

    result = validate(answer)

    expected_missing = [
        r.t
        for r in result.records
        if (r.word_position // width == line_number - 1 and r.word_position % width == width - 1)
        or r.word_position // width == line_number
    ]
    assert [r.t for r in result.records if r.z == 0] == expected_missing
    assert all(r.outcome == "missing" for r in result.records if r.z == 0)
    assert result.first_fail_kind_3_line == line_number
    assert result.first_fail_kind_2_line == line_number + 1
    assert result.broken_line_count == 2


def test_glued_lines_in_the_middle_are_kind_3(task, reference, validate) -> None:
    """Glued lines in the middle of the text: the number of lines did not match - the 3rd kind.

    The shifted lines are recognizable by their word count, so absorption stays silent; the
    loss of control is betrayed by the missing line. The failure is recorded on line 2, where
    the glue starts, and not at the end: line 1 is the last exact copy at its own number, and
    the next exact copy - reference line 4 - stands at number 3.
    """
    width = reference.words_per_line
    line_count = reference.text.count("\n") + 1
    assert line_count >= 4
    answer = glue_lines(reference, 2)
    assert answer.count("\n") + 1 == line_count - 1

    result = validate(answer)

    assert result.first_fail_kind_3_line == 2
    assert result.first_fail_kind_2_line == 2
    assert result.broken_line_count == 2
    # Line 1 is intact; the glued line 2 starts with its reference content, and its trials
    # are intact; the last reference line is missing - its trials are unreachable. The
    # shifted lines between them are the price of the conservative scheme (section 7 of the
    # plan): their verdicts depend on random word coincidences and are not fixed here.
    assert any(r.word_position // width == line_count - 1 for r in result.records)
    for record in result.records:
        row = record.word_position // width
        if row <= 1:
            assert record.z == 1
            assert record.outcome == "correct"
        if row == line_count - 1:
            assert record.outcome == "missing"
    assert result.p_strict is False


def test_empty_answer_zeroes_everything_without_exception(task, reference, validate) -> None:
    """An empty answer is a valid measurement result: all trials are unreachable."""
    line_count = reference.text.count("\n") + 1

    result = validate("")

    assert result.h == task.source_text.replacement_count
    assert all(r.z == 0 and r.outcome == "missing" for r in result.records)
    assert result.p_event == 0.0
    assert result.err_rate == 1.0
    assert result.first_error_t == 1
    assert result.first_fail_kind_3_line == 1
    assert result.first_fail_kind_2_line == line_count
    assert result.broken_line_count == line_count
    assert result.background_errors == 0
    assert result.p_strict is False


# --- Zero misses: the answer wrapper (a form violation, not a degradation of replacements) ---

# Model remarks: the width of each line differs from the reference one (10 words) - a line of
# full width is indistinguishable from the protocol word stream and so is never stripped.
PREAMBLE = "Here is the rewritten text:"
POSTSCRIPT = "Let me know if you need anything else!"
LOWERCASE_PREAMBLE = "here is the rewritten text"


def wrap(answer: str, *, before: tuple[str, ...] = (), after: tuple[str, ...] = ()) -> str:
    """Frame the answer with model remarks - form damage without damage to the useful text."""
    return "\n".join([*before, answer, *after])


@pytest.mark.parametrize(
    ("before", "after", "leading", "trailing"),
    [
        ((PREAMBLE,), (), 1, 0),
        ((), (POSTSCRIPT,), 0, 1),
        ((PREAMBLE,), (POSTSCRIPT,), 1, 1),
        (("```",), ("```",), 1, 1),
        ((PREAMBLE, ""), ("", POSTSCRIPT), 2, 2),
        ((LOWERCASE_PREAMBLE,), (), 1, 0),
    ],
)
def test_wrapper_is_stripped_and_counted_exactly(
    reference, validate, before, after, leading, trailing
) -> None:
    """The wrapper is stripped, the core is measured fully, the form violation is counted.

    A perfectly executed task in a wrapper must give all ``z = 1`` and an empty failure
    ladder: otherwise the instrument would measure the manners of the model instead of its
    capacity. The strict metric stays false - the instruction demanded only the rewritten
    text.
    """
    result = validate(wrap(reference.text, before=before, after=after))

    assert all(record.z == 1 for record in result.records)
    assert result.p_event == 1.0
    assert result.wrapper_leading_lines == leading
    assert result.wrapper_trailing_lines == trailing
    assert result.wrapper_stripped is True
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0
    assert result.background_errors == 0
    assert result.p_strict is False


def test_wrapper_and_a_lost_replacement_are_measured_independently(
    task, reference, validate
) -> None:
    """Extraction and measurement do not interfere: the wrapper is stripped, the miss found."""
    t = 4
    position = _trial_positions(task)[t - 1]
    damaged = replace_word(reference, position, task.source_text.words[position])

    result = validate(wrap(damaged, before=(PREAMBLE,), after=(POSTSCRIPT,)))

    assert result.wrapper_stripped is True
    assert result.wrapper_leading_lines == 1
    assert result.wrapper_trailing_lines == 1
    assert [r.t for r in result.records if r.z == 0] == [t]
    assert result.records[t - 1].outcome == "unreplaced"
    assert result.first_fail_kind_2_line is None
    assert result.first_fail_kind_3_line is None
    assert result.broken_line_count == 0


def test_lowercase_wrapper_survives_an_unreplaced_edge_word(task, reference, validate) -> None:
    """The stage 2 anchor is the pair {replacement, target}: a miss in the first position holds.

    Stage 1 is powerless here (a wrapper of lowercase words), and the first word of the core
    is not the reference one: the second branch of the anchor works, otherwise the most
    frequent model error would switch extraction off.
    """
    first_position = _trial_positions(task)[0]
    damaged = replace_word(reference, first_position, task.source_text.words[first_position])

    result = validate(wrap(damaged, before=(LOWERCASE_PREAMBLE,)))

    assert result.wrapper_stripped is True
    assert result.wrapper_leading_lines == 1
    assert result.records[0].outcome == "unreplaced"
    assert [r.t for r in result.records if r.z == 0] == [1]
    assert result.first_fail_kind_3_line is None


def test_lowercase_wrapper_with_a_broken_edge_word_falls_back(
    task, vocab, reference, validate
) -> None:
    """The honest boundary of the scheme: both stages are silent - no extraction, 3rd kind.

    A lowercase wrapper is opaque to the alphabet, and a foreign word in the first position
    of the **stream** breaks the start anchor. The instrument does not guess: the wrapper
    counters are zero, the extra line is honestly recorded as a loss of control over lines.
    """
    del task
    assert FOREIGN_WORD not in vocab.words
    damaged = replace_word(reference, 0, FOREIGN_WORD)

    result = validate(wrap(damaged, before=(LOWERCASE_PREAMBLE,)))

    assert result.wrapper_stripped is False
    assert result.wrapper_leading_lines == 0
    assert result.wrapper_trailing_lines == 0
    assert result.first_fail_kind_3_line is not None


def test_an_appended_line_of_words_is_not_mistaken_for_a_wrapper(reference, validate) -> None:
    """An appended word stream is not a wrapper: stripping would hide a loss of text control."""
    width = reference.words_per_line
    line_count = reference.text.count("\n") + 1
    appended = " ".join([FOREIGN_WORD] * width)

    result = validate(wrap(reference.text, after=(appended,)))

    assert result.wrapper_stripped is False
    assert result.wrapper_trailing_lines == 0
    assert result.first_fail_kind_3_line == line_count + 1
    assert result.p_event == 1.0


# --- End-to-end run: consistency of all links on one task ---


def test_end_to_end_chain_is_consistent(task, vocab, reference, validate) -> None:
    """Generation -> reference -> mutation -> validator -> statistics: links agree.

    The records are bound to the markup and the table of the task, the aggregates agree with
    the records, the ``as_dict()`` snapshot survives a JSON round trip.
    """
    t = 5
    position = _trial_positions(task)[t - 1]
    assert FOREIGN_WORD not in vocab.words

    result = validate(replace_word(reference, position, FOREIGN_WORD))

    assert result.h == task.source_text.replacement_count
    assert [r.t for r in result.records] == list(range(1, result.h + 1))
    assert [r.word_position for r in result.records] == _trial_positions(task)
    assert all(r.expected == task.rule_table.targets[r.rule_index - 1] for r in result.records)
    assert sum(stat.attempts for stat in result.rule_stats) == result.h
    assert sum(stat.correct for stat in result.rule_stats) == result.correct_count
    assert result.correct_count == result.h - 1

    restored = json.loads(json.dumps(result.as_dict()))
    assert restored["h"] == result.h
    assert restored["p_event"] == result.p_event
    assert restored["records"][t - 1]["outcome"] == "other_word"
    assert restored["first_error_t"] == t
