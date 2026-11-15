"""The alignment of an answer by places: anchors, gaps, glue and split, wrapper lines, places."""

from __future__ import annotations

from collections import Counter

from iec.answer_validator import build_reference
from iec.answer_validator.alignment.aligned_answer import (
    GROUP_ANCHOR,
    PAIRING_GLUE,
    PAIRING_ONE,
    PAIRING_SPLIT,
    PLACE_DELETED,
    PLACE_INSERTED,
    PLACE_MATCHED,
    AlignedAnswer,
    Place,
)
from iec.answer_validator.alignment.line_alignment import align_answer
from iec.contracts import Reference, Rule, RuleTable, SourceText
from iec.prompt_generator.text_generator import WORDS_PER_LINE

RULES = (("lake", "stone"), ("tree", "fish"))


def _source_line(index: int) -> list[str]:
    """A source line: eight background words of its own, lake at index, tree at index + 5.

    The background words of a line occur in no other line, so a foreign line matches a line in
    at most two words (lake and tree), and a copy is easy to tell.
    """
    words = [f"{'abcdefgh'[index]}{'abcdefghij'[place]}q" for place in range(10)]
    words[index % 10] = "lake"
    words[(index + 5) % 10] = "tree"
    return words


SOURCE = [_source_line(index) for index in range(8)]
REFERENCE = [
    ["stone" if word == "lake" else "fish" if word == "tree" else word for word in line]
    for line in SOURCE
]


def _task() -> tuple[RuleTable, SourceText, Reference]:
    """The task of the tests, built from the contracts as a published record gives it."""
    table = RuleTable(
        rules=tuple(Rule(source=source, target=target) for source, target in RULES),
        vocab_id="v1_intersection",
        seed_rules=1,
    )
    words = tuple(word for line in SOURCE for word in line)
    sources = frozenset(table.sources)
    text = SourceText(
        words=words,
        words_per_line=WORDS_PER_LINE,
        replacement_positions=tuple(index for index, word in enumerate(words) if word in sources),
        len_tokens=0,
        tokenizer_name="unmeasured",
        seed_text=2,
    )
    return table, text, build_reference(table, text)


def _align(lines: list[list[str]]) -> AlignedAnswer:
    table, text, reference = _task()
    return align_answer("\n".join(" ".join(line) for line in lines), reference, text, table)


def _lines(*numbers: int) -> list[list[str]]:
    return [list(REFERENCE[number]) for number in numbers]


def _kinds(aligned: AlignedAnswer) -> Counter[str]:
    return Counter(place.kind for place in aligned.places)


def _words(aligned: AlignedAnswer) -> dict[int, str | None]:
    return {
        place.reference_position: place.word
        for place in aligned.places
        if place.reference_position is not None
    }


def _answer_positions(aligned: AlignedAnswer, kind: str, low: int, high: int) -> list[int]:
    """Answer word numbers of the places of ``kind`` with a reference number in ``[low, high)``;
    for insertions - all answer word numbers if ``low < 0``."""
    found = []
    for place in aligned.places:
        if place.kind != kind or place.answer_position is None:
            continue
        if low < 0 or (
            place.reference_position is not None and low <= place.reference_position < high
        ):
            found.append(place.answer_position)
    return sorted(found)


def test_answer_equal_to_reference_is_all_matched() -> None:
    aligned = _align(_lines(*range(8)))

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    assert (aligned.answer_words, aligned.wrapper_lines, aligned.episodes) == (80, 0, 0)
    assert aligned.pairings == ()
    assert all(place.answer_position == place.reference_position for place in aligned.places)


def test_misfire_wrong_rule_and_unreplaced_stay_on_their_places() -> None:
    lines = _lines(*range(8))
    lines[2][1] = "fish"  # a misfire on a background word of line 3
    lines[5][5] = "fish"  # on the place of lake in line 6 - the right-hand word of another rule
    lines[3][3] = "lake"  # the rule of line 4 is not applied: the line stays an anchor

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    words = _words(aligned)
    assert (words[21], words[55], words[33]) == ("fish", "fish", "lake")
    assert aligned.pairings == ((PAIRING_ONE, 0.9), (PAIRING_ONE, 0.9))
    assert aligned.episodes == 0


def test_lost_word_costs_one_deleted_place_and_the_rest_of_the_line_is_checked() -> None:
    lines = _lines(*range(8))
    del lines[4][8]  # a background word of line 5

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 79, PLACE_DELETED: 1})
    assert [p.reference_position for p in aligned.places if p.kind == PLACE_DELETED] == [48]
    assert _words(aligned)[49] == "fish"  # the replacement after the lost word is checked
    assert aligned.episodes == 1


def test_extra_word_costs_one_inserted_place() -> None:
    lines = _lines(*range(8))
    lines[4].insert(3, lines[4][3])

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80, PLACE_INSERTED: 1})


def test_return_back_matches_the_first_pass_and_inserts_the_repeat() -> None:
    aligned = _align(_lines(0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7))

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80, PLACE_INSERTED: 30})
    assert _answer_positions(aligned, PLACE_MATCHED, 10, 40) == list(range(10, 40))
    assert _answer_positions(aligned, PLACE_INSERTED, -1, 0) == list(range(40, 70))
    assert aligned.episodes == 1


def test_return_back_without_continuation_inserts_the_repeat_and_deletes_the_rest() -> None:
    aligned = _align(_lines(0, 1, 2, 3, 1, 2))

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 40, PLACE_INSERTED: 20, PLACE_DELETED: 40})


def test_jump_forward_deletes_the_skipped_lines() -> None:
    aligned = _align(_lines(0, 1, 4, 5, 6, 7))

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 60, PLACE_DELETED: 20})
    deleted = sorted(
        p.reference_position
        for p in aligned.places
        if p.kind == PLACE_DELETED and p.reference_position is not None
    )
    assert deleted == list(range(20, 40))


def test_truncated_answer_keeps_the_written_start_of_the_last_line() -> None:
    lines = _lines(0, 1, 2)
    lines.append(list(REFERENCE[3][:4]))

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 34, PLACE_DELETED: 46})
    assert [kind for kind, _ in aligned.pairings] == [PAIRING_ONE]


def test_loop_of_one_word_is_inserted() -> None:
    lines = _lines(0, 1, 2) + [["abq"] * 10 for _ in range(5)]

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 30, PLACE_INSERTED: 50, PLACE_DELETED: 50})
    assert aligned.pairings == ()


def test_glued_lines_without_lost_words_are_all_matched() -> None:
    lines = _lines(0, 1, 2) + [REFERENCE[3] + REFERENCE[4]] + _lines(5, 6, 7)

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    assert [kind for kind, _ in aligned.pairings] == [PAIRING_GLUE]
    assert aligned.episodes == 0


def test_glue_with_lost_words_matches_what_was_written() -> None:
    glued = REFERENCE[3][:2] + REFERENCE[4][2:]
    lines = _lines(0, 1, 2) + [glued] + _lines(5, 6, 7)

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 70, PLACE_DELETED: 10})
    assert [kind for kind, _ in aligned.pairings] == [PAIRING_GLUE]


def test_split_line_is_all_matched() -> None:
    lines = _lines(0, 1, 2) + [REFERENCE[3][:6], REFERENCE[3][6:]] + _lines(4, 5, 6, 7)

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    assert [kind for kind, _ in aligned.pairings] == [PAIRING_SPLIT]


def test_swapped_neighbour_lines_cost_twenty_places() -> None:
    aligned = _align(_lines(0, 1, 3, 2, 4, 5, 6, 7))

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 70, PLACE_DELETED: 10, PLACE_INSERTED: 10})
    assert aligned.episodes == 2


def test_wrapper_lines_are_not_places() -> None:
    lines = [
        ["Here", "is", "the", "rewritten", "text:"],
        *_lines(*range(8)),
        ["Let", "me", "know."],
    ]

    aligned = _align(lines)

    assert aligned.wrapper_lines == 2
    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    assert aligned.answer_words == 80


def test_empty_answer_deletes_every_reference_word() -> None:
    table, text, reference = _task()

    aligned = align_answer("", reference, text, table)

    assert _kinds(aligned) == Counter({PLACE_DELETED: 80})
    assert (aligned.wrapper_lines, aligned.answer_words) == (1, 0)


def test_swapped_neighbour_words_are_two_wrong_words_on_their_places() -> None:
    lines = _lines(*range(8))
    lines[6][2], lines[6][3] = lines[6][3], lines[6][2]

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 80})
    words = _words(aligned)
    assert (words[62], words[63]) == (REFERENCE[6][3], REFERENCE[6][2])


def test_line_with_most_background_words_misfired_is_not_a_copy() -> None:
    lines = _lines(*range(8))
    for place in (0, 2, 3, 4, 5, 7):  # six of the eight background words of line 7
        lines[6][place] = "fish"

    aligned = _align(lines)

    assert _kinds(aligned) == Counter({PLACE_MATCHED: 70, PLACE_DELETED: 10, PLACE_INSERTED: 10})


def test_core_line_sizes_leave_out_the_wrapper() -> None:
    aligned = _align([["Here", "it", "is:"], *_lines(0, 1), ["Done."]])
    assert aligned.wrapper_lines == 2
    assert aligned.core_line_sizes.tolist() == [10, 10]


def test_places_are_plain_values_in_the_order_of_the_arrays() -> None:
    aligned = _align(_lines(0, 2))
    places = aligned.places
    assert len(places) == len(aligned.kinds)
    assert places[0] == Place(PLACE_MATCHED, 0, 0, REFERENCE[0][0], GROUP_ANCHOR)
    deleted = next(place for place in places if place.kind == PLACE_DELETED)
    assert (deleted.answer_position, deleted.word) == (None, None)
    assert type(deleted.reference_position) is int
