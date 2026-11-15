"""Mirror tests of the word matrices: the subsequence of many pairs at once."""

import random

import numpy as np

from iec.answer_validator.alignment.gaps.small_gap import lcs_length
from iec.answer_validator.alignment.gaps.word_matrices import encode_lines, lcs_many, word_masks


def test_the_subsequence_of_many_pairs_equals_the_plain_one() -> None:
    chance = random.Random(7)
    alphabet = [f"w{number}" for number in range(6)]
    pairs = [
        (
            tuple(chance.choices(alphabet, k=chance.randint(0, 9))),
            tuple(chance.choices(alphabet, k=chance.randint(1, 20))),
        )
        for _ in range(500)
    ]
    ids: dict[str, int] = {}
    reference = encode_lines([right for _, right in pairs], ids)
    answer = encode_lines([left for left, _ in pairs], ids)
    numbers = np.arange(len(pairs))
    counts = lcs_many(answer.flat, answer.starts, answer.sizes, word_masks(reference), numbers)
    assert counts.tolist() == [lcs_length(left, right) for left, right in pairs]


def test_the_subsequence_of_a_glue_counts_two_lines_as_one_run() -> None:
    chance = random.Random(8)
    alphabet = [f"w{number}" for number in range(6)]
    lines = [tuple(chance.choices(alphabet, k=chance.randint(1, 32))) for _ in range(40)]
    answers = [tuple(chance.choices(alphabet, k=chance.randint(0, 40))) for _ in range(39)]
    ids: dict[str, int] = {}
    reference = encode_lines(lines, ids)
    answer = encode_lines(answers, ids)
    numbers = np.arange(len(answers))
    counts = lcs_many(
        answer.flat,
        answer.starts,
        answer.sizes,
        word_masks(reference),
        numbers,
        reference.sizes[numbers],
    )
    assert counts.tolist() == [
        lcs_length(answers[t], lines[t] + lines[t + 1]) for t in range(len(answers))
    ]
