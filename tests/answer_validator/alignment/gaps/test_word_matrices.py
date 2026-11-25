"""Mirror tests of the word matrices: the subsequence of many pairs at once."""

import random

import numpy as np

from iec.answer_validator.alignment.gaps.small_gap import lcs_length
from iec.answer_validator.alignment.gaps.word_matrices import (
    _lcs_pairs,
    _lcs_steps,
    _shared_merge,
    _shared_steps,
    encode_lines,
    lcs_many,
    word_counts,
    word_masks,
)


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


def test_the_compiled_and_the_numpy_subsequences_are_equal() -> None:
    rng = np.random.default_rng(11)
    ids: dict[str, int] = {}
    reference = encode_lines(
        [
            tuple(f"w{int(x)}" for x in rng.integers(0, 12, size=int(rng.integers(1, 9))))
            for _ in range(20)
        ],
        ids,
    )
    answer = encode_lines(
        [
            tuple(f"w{int(x)}" for x in rng.integers(0, 14, size=int(rng.integers(1, 30))))
            for _ in range(50)
        ],
        ids,
    )
    masks = word_masks(reference)
    pairs = 400
    lines = rng.integers(0, 50, size=pairs)
    targets = rng.integers(0, 19, size=pairs)
    starts, sizes = answer.starts[lines], answer.sizes[lines]
    for first in (None, reference.sizes[targets]):
        steps = _lcs_steps(answer.flat, starts, sizes, masks, targets, first)
        compiled_pairs = _lcs_pairs(
            answer.flat.astype(np.int64),
            starts.astype(np.int64),
            sizes.astype(np.int64),
            masks,
            targets.astype(np.int64),
            (np.zeros(1, np.int64) if first is None else first).astype(np.int64),
            first is not None,
        )
        assert np.array_equal(steps, compiled_pairs)


def test_the_compiled_and_the_numpy_common_words_are_equal() -> None:
    rng = np.random.default_rng(12)
    ids: dict[str, int] = {}
    reference = encode_lines(
        [tuple(f"w{int(x)}" for x in rng.integers(0, 12, size=8)) for _ in range(15)], ids
    )
    answer = encode_lines(
        [tuple(f"w{int(x)}" for x in rng.integers(0, 14, size=10)) for _ in range(25)], ids
    )
    left, right = word_counts(answer), word_counts(reference)
    arguments = (*left, *right, len(answer.sizes), len(reference.sizes))
    assert np.array_equal(_shared_steps(*arguments), _shared_merge(*arguments))
