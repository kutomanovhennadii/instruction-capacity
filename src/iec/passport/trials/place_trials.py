"""Trials of the three quantities of the formula by the places of the alignment.

``P_correct = P_sync · [π · H + (1 − π) · (1 − F)]``. A trial of P_sync is a place of the
alignment: a success - the answer word stands against its reference word, a failure - a deletion
or an insertion. A trial of H is a rule word standing on its place: a success - the right
replacement; the choice of the rule is absorbed by H. A trial of F is a background word standing
on its place: the event is a misfire, the right-hand word of a rule on the place of a background
word.

The word number s is counted by the reference, as in the passport: for a reference word - its
number from one; for an insertion - the number of the reference word that follows the last
matched word standing in the answer before it (1 before the first matched word), but not past
the end of the text. The word-number bin is the doubling of the line number
``doub_bin(⌈s/10⌉)``, as in the passport.

The places are counted by whole arrays. The doubling bins of the lines and the logarithms of the
word numbers are read from the tables of the measurement (``trial_tables``), built once at its
setup. The sums of the logarithms go in the order of the places (``np.add.at``), so they are equal
to a sum in a loop to the last bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iec.answer_validator import (
    KIND_DELETED,
    KIND_MATCHED,
    PLACE_DELETED,
    PLACE_INSERTED,
    PLACE_MATCHED,
    AlignedAnswer,
)
from iec.contracts import Reference, RuleTable, SourceText
from iec.passport.trials.trial_tables import TrialTables

#: The counts of a point in the order of a ``points.jsonl`` row.
TOTALS: tuple[str, ...] = (
    "places",
    "matched",
    "deleted",
    "inserted",
    "wrapper_lines",
    "episodes",
    "rule_trials",
    "rule_correct",
    "rule_unreplaced",
    "rule_wrong_rule",
    "rule_other_word",
    "rule_deleted",
    "background_trials",
    "background_misfires",
    "background_other_change",
    "background_deleted",
    "pairings_one",
    "pairings_glue",
    "pairings_split",
)


@dataclass(frozen=True, slots=True)
class PointTrials:
    """The trials of one point by word-number bins.

    ``sync`` - ``(s bin, places, failures, sum of log₂ s)``; ``rule`` - ``(s bin, rule number,
    trials, failures, sum of log₂ s)``; ``background`` - ``(s bin, trials, misfires, sum of
    log₂ s)``. A failure of H is any word but the right replacement; the event of F is a
    misfire. ``totals`` - the counts of the point by :data:`TOTALS`.
    """

    key: str
    D: int  # noqa: N815 - protocol axis name
    n: int
    m: int
    sync: tuple[tuple[int, int, int, float], ...]
    rule: tuple[tuple[int, int, int, int, float], ...]
    background: tuple[tuple[int, int, int, float], ...]
    totals: dict[str, int]


def _fold_trials(
    keys: np.ndarray, failures: np.ndarray, logarithms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fold trials by keys: sorted keys, trials, failures and sums of logarithms in place order."""
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, logarithms)
    return unique, counts, np.bincount(inverse[failures], minlength=len(unique)), sums


def point_trials(
    key: str,
    D: int,  # noqa: N803 - protocol axis name
    n: int,
    m: int,
    aligned: AlignedAnswer,
    reference: Reference,
    source_text: SourceText,
    rule_table: RuleTable,
    *,
    tables: TrialTables,
) -> PointTrials:
    """Split the places of a point into trials of P_sync, H and F by word-number bins.

    Args:
        key: The key of the point.
        D: The number of distinct words of the text.
        n: How many times every word occurs in the text.
        m: The number of rules.
        aligned: The alignment.
        reference: The reference.
        source_text: The source text of the task.
        rule_table: The rule table in the order shown to the model.
        tables: The tables of the measurement (``build_trial_tables``); they must cover the
            length of the text of the point.

    Returns:
        ``PointTrials``.

    Raises:
        ContractViolationError: ``TRIAL_TABLES_INVALID`` - the tables are shorter than the text
            of the point.
    """
    width = reference.words_per_line
    size = len(reference.words)
    kinds = aligned.kinds
    positions = aligned.reference_positions
    answers = aligned.answer_positions

    numbers = positions + 1
    inserted = positions < 0
    if inserted.any():
        matched_places = np.flatnonzero(kinds == KIND_MATCHED)
        order = np.argsort(answers[matched_places])
        answer_sorted = answers[matched_places][order]
        reference_sorted = positions[matched_places][order]
        slots = np.searchsorted(answer_sorted, answers[inserted]) - 1
        following = np.ones(len(slots), dtype=np.int64)
        known = slots >= 0
        following[known] = reference_sorted[slots[known]] + 2
        numbers[inserted] = np.minimum(following, size)
    tables.require(int(numbers.max(initial=0)))
    lines = (numbers - 1) // width + 1
    s_bins = tables.line_bins[lines]
    logarithms = tables.logarithms[numbers]

    totals = dict.fromkeys(TOTALS, 0)
    totals["wrapper_lines"] = aligned.wrapper_lines
    totals["episodes"] = aligned.episodes
    for kind, _share in aligned.pairings:
        totals[f"pairings_{kind}"] += 1
    totals["places"] = len(kinds)
    totals[PLACE_MATCHED] = int(np.count_nonzero(kinds == KIND_MATCHED))
    totals[PLACE_DELETED] = int(np.count_nonzero(kinds == KIND_DELETED))
    totals[PLACE_INSERTED] = totals["places"] - totals[PLACE_MATCHED] - totals[PLACE_DELETED]

    sync_bins, sync_places, sync_failures, sync_sums = _fold_trials(
        s_bins, kinds != KIND_MATCHED, logarithms
    )

    targets = tuple(rule_table.targets)
    sources = tuple(rule.source for rule in rule_table.rules)
    ids = {
        word: number
        for number, word in enumerate(
            {*reference.words, *source_text.words, *aligned.core_words, *targets, *sources}
        )
    }

    def encoded(words: tuple[str, ...]) -> np.ndarray:
        """Return the numbers of the words by the dictionary of the point."""
        return np.fromiter(map(ids.__getitem__, words), dtype=np.int64, count=len(words))

    expected_words = encoded(reference.words)
    source_words = encoded(source_text.words)
    core_words = encoded(aligned.core_words)
    target_ids = encoded(targets)
    rule_ids = encoded(sources)
    is_target = np.zeros(len(ids), dtype=bool)
    is_target[target_ids] = True
    rule_numbers = np.zeros(len(ids), dtype=np.int64)
    rule_numbers[rule_ids] = np.arange(1, len(rule_ids) + 1)
    marked = np.zeros(size, dtype=bool)
    marked[np.fromiter(source_text.replacement_positions, dtype=np.int64)] = True

    on_reference = np.flatnonzero(positions >= 0)
    place_positions = positions[on_reference]
    deleted = kinds[on_reference] == KIND_DELETED
    rule_place = marked[place_positions]
    totals["rule_deleted"] = int(np.count_nonzero(rule_place & deleted))
    totals["background_deleted"] = int(np.count_nonzero(~rule_place & deleted))

    standing = on_reference[~deleted]
    standing_positions = positions[standing]
    standing_rule = marked[standing_positions]
    words = core_words[answers[standing]]

    rule_places = standing[standing_rule]
    rule_positions = standing_positions[standing_rule]
    rule_words = words[standing_rule]
    correct = rule_words == expected_words[rule_positions]
    unreplaced = ~correct & (rule_words == source_words[rule_positions])
    wrong_rule = ~correct & ~unreplaced & is_target[rule_words]
    totals["rule_trials"] = len(rule_places)
    totals["rule_correct"] = int(np.count_nonzero(correct))
    totals["rule_unreplaced"] = int(np.count_nonzero(unreplaced))
    totals["rule_wrong_rule"] = int(np.count_nonzero(wrong_rule))
    totals["rule_other_word"] = (
        totals["rule_trials"]
        - totals["rule_correct"]
        - totals["rule_unreplaced"]
        - totals["rule_wrong_rule"]
    )
    index_base = len(rule_table.rules) + 1
    rule_keys = s_bins[rule_places] * index_base + rule_numbers[source_words[rule_positions]]
    rule_cells, rule_trials, rule_failures, rule_sums = _fold_trials(
        rule_keys, ~correct, logarithms[rule_places]
    )

    background_places = standing[~standing_rule]
    background_positions = standing_positions[~standing_rule]
    background_words = words[~standing_rule]
    misfire = is_target[background_words]
    totals["background_trials"] = len(background_places)
    totals["background_misfires"] = int(np.count_nonzero(misfire))
    totals["background_other_change"] = int(
        np.count_nonzero(~misfire & (background_words != expected_words[background_positions]))
    )
    background_bins, background_trials, background_misfires, background_sums = _fold_trials(
        s_bins[background_places], misfire, logarithms[background_places]
    )

    return PointTrials(
        key=key,
        D=D,
        n=n,
        m=m,
        sync=tuple(
            zip(
                sync_bins.tolist(),
                sync_places.tolist(),
                sync_failures.tolist(),
                sync_sums.tolist(),
                strict=True,
            )
        ),
        rule=tuple(
            zip(
                (rule_cells // index_base).tolist(),
                (rule_cells % index_base).tolist(),
                rule_trials.tolist(),
                rule_failures.tolist(),
                rule_sums.tolist(),
                strict=True,
            )
        ),
        background=tuple(
            zip(
                background_bins.tolist(),
                background_trials.tolist(),
                background_misfires.tolist(),
                background_sums.tolist(),
                strict=True,
            )
        ),
        totals=totals,
    )
