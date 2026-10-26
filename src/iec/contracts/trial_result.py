"""Validation result of one point: the trials ``z_1 ... z_H`` and statistics over them."""

from __future__ import annotations

from dataclasses import dataclass

OUTCOME_CORRECT = "correct"
OUTCOME_UNREPLACED = "unreplaced"
OUTCOME_WRONG_RULE_TARGET = "wrong_rule_target"
OUTCOME_OTHER_WORD = "other_word"
OUTCOME_MISSING = "missing"

OUTCOMES: tuple[str, ...] = (
    OUTCOME_CORRECT,
    OUTCOME_UNREPLACED,
    OUTCOME_WRONG_RULE_TARGET,
    OUTCOME_OTHER_WORD,
    OUTCOME_MISSING,
)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One binary trial - one occurrence of a word to replace in the text.

    The full set of fields is set by the needs of the analysis: without ``t`` the
    degradation along the text cannot be built, without ``rule_index`` the positional cost
    of a rule cannot. The verdict ``z`` is strictly binary; ``outcome`` is the category of
    what happened (a value from :data:`OUTCOMES`); it does not take part in ``z`` but tells
    apart the mechanisms of degradation ("rule lost" against "rules mixed up"), which cannot
    be restored later. ``observed`` is ``None`` when the position is unreachable (an
    earlier structural collapse).
    """

    t: int
    rule_index: int
    word_position: int
    expected: str
    observed: str | None
    z: int
    outcome: str

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the trial."""
        return {
            "t": self.t,
            "rule_index": self.rule_index,
            "word_position": self.word_position,
            "expected": self.expected,
            "observed": self.observed,
            "z": self.z,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class RuleStat:
    """Statistics of one rule: how many of its replacements the model did right.

    ``attempts`` is the number of trials of the rule in the task (equal to ``v/D`` by
    construction), ``p_rule`` is the share of correct replacements; it is the stability of
    applying the rule and the raw material of the positional cost.
    """

    rule_index: int
    attempts: int
    correct: int

    @property
    def p_rule(self) -> float:
        """Return the share of correct replacements of the rule (0.0 without trials)."""
        if self.attempts == 0:
            return 0.0
        return self.correct / self.attempts

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the rule statistics."""
        return {
            "rule_index": self.rule_index,
            "attempts": self.attempts,
            "correct": self.correct,
            "p_rule": self.p_rule,
        }


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Summary of one point: the full trial records and the statistics of the point.

    ``records`` is the full raw material in ``t`` order (exactly ``H`` records); aggregates
    that follow from it unambiguously (:attr:`h`, :attr:`correct_count`, :attr:`p_event`,
    :attr:`err_rate`) are declared as properties, not fields: a stored derivative can
    diverge from its source. Only facts not derivable from ``records`` are stored - the
    collapse ladder, line counters, background and the strict metric. ``rule_stats`` are in
    the order of the replacement table (``rule_index`` 1..m).

    The ladder of three kinds of collapse by severity:

    * **collapse of the 1st kind** - a wrong replacement with the text structure intact:
      a rule is lost or mixed up, the structure is untouched. It is recorded by the trial
      itself (``z = 0``); the first case is :attr:`first_error_t` and the rule that
      stumbled;
    * **collapse of the 2nd kind** - a structure violation absorbed by one line: the number
      of words of the line does not match the reference, but the next line is recognizable,
      and synchronization on its anchor is restored (:attr:`first_fail_kind_2_line`). The
      price is the tail of one line;
    * **collapse of the 3rd kind** - loss of control over lines: the damage is not absorbed
      (the next line is damaged too - this includes words flowing between lines) or the
      number of lines does not match. :attr:`first_fail_kind_3_line` is the reference line
      where the answer loses synchronization with its text (by exact copies of lines,
      forward through the text), never later than the first unabsorbed damaged line and the
      first line beyond the end of the shorter side.

    The steps are independent: an isolated collapse of the 2nd kind in an early line and a
    collapse of the 3rd kind further on coexist in one result.

    :attr:`broken_line_count` is the scale of structure damage: how many answer lines have a
    word count different from the reference or are missing altogether. Wrong replacements
    are not counted here - they do not break the structure and are accounted for in
    ``records`` and ``rule_stats``.

    The ``wrapper_*`` fields record a **violation of the answer form** - framing the useful
    text with the model's own remarks ("Here is the rewritten text", a code block fence),
    forbidden by the prompt instruction. This is a fact independent of the ability to keep
    the rule table, so it is accounted for separately: a stripped wrapper does not cancel a
    single trial, and the trials are evaluated on the core of the answer. These fields do
    not enter the verdicts ``z``; :attr:`p_strict` excludes a wrapper - the instruction asked
    for the rewritten text only. The wrapper text itself is not stored here: the raw answer
    is kept whole in the point record, a duplicate would be a second source of truth.

    The sign "did not happen" is ``None`` everywhere, not ``0``: for a collapse of the 1st
    kind this is a censored observation, and the analysis must tell "no errors" from "an
    error at the end".

    .. note::
       "Kind of collapse" is a severity scale of answer damage and has nothing to do with
       type I and type II errors in statistical hypothesis testing.
    """

    records: tuple[TrialRecord, ...]
    rule_stats: tuple[RuleStat, ...]
    first_error_t: int | None
    first_error_rule_index: int | None
    first_fail_kind_2_line: int | None
    first_fail_kind_3_line: int | None
    broken_line_count: int
    background_errors: int
    p_strict: bool
    wrapper_leading_lines: int = 0
    wrapper_trailing_lines: int = 0
    wrapper_stripped: bool = False

    @property
    def h(self) -> int:
        """Return the number of trials of the point ``H``."""
        return len(self.records)

    @property
    def correct_count(self) -> int:
        """Return the number of correct replacements (the sum of ``z``)."""
        return sum(record.z for record in self.records)

    @property
    def p_event(self) -> float:
        """Return the per-event share of correct replacements (0.0 without trials)."""
        if self.h == 0:
            return 0.0
        return self.correct_count / self.h

    @property
    def err_rate(self) -> float:
        """Return the per-event share of errors ``1 - p_event``."""
        return 1.0 - self.p_event

    def as_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the point with records and aggregates."""
        return {
            "records": [record.as_dict() for record in self.records],
            "rule_stats": [stat.as_dict() for stat in self.rule_stats],
            "h": self.h,
            "correct_count": self.correct_count,
            "p_event": self.p_event,
            "err_rate": self.err_rate,
            "first_error_t": self.first_error_t,
            "first_error_rule_index": self.first_error_rule_index,
            "first_fail_kind_2_line": self.first_fail_kind_2_line,
            "first_fail_kind_3_line": self.first_fail_kind_3_line,
            "broken_line_count": self.broken_line_count,
            "background_errors": self.background_errors,
            "p_strict": self.p_strict,
            "wrapper_leading_lines": self.wrapper_leading_lines,
            "wrapper_trailing_lines": self.wrapper_trailing_lines,
            "wrapper_stripped": self.wrapper_stripped,
        }
