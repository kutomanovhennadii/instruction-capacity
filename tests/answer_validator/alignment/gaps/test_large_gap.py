"""Mirror tests of the large gap: on random gaps it decides as the small one."""

from __future__ import annotations

import random

import pytest

from iec.answer_validator.alignment.gaps.large_gap import LargeGapResolver
from iec.answer_validator.alignment.gaps.small_gap import SmallGapResolver


def _random_gap(
    seed: int,
) -> tuple[list[int], list[int], list[tuple[str, ...]], list[tuple[str, ...]], frozenset]:
    """A random gap over a few words, so that glues, splits and ties are frequent."""
    chance = random.Random(seed)
    alphabet = [f"w{number}" for number in range(chance.randint(2, 12))]
    reference = [
        tuple(chance.choices(alphabet, k=chance.randint(1, 6)))
        for _ in range(chance.randint(1, 9))
    ]
    answer: list[tuple[str, ...]] = []
    for _ in range(chance.randint(1, 14)):
        draw = chance.random()
        if draw < 0.1:
            answer.append(())
        elif draw < 0.2:
            answer.append(chance.choice(reference))
        elif draw < 0.6:
            line = list(chance.choice(reference))
            line[chance.randrange(len(line))] = chance.choice(alphabet)
            answer.append(tuple(line))
        else:
            answer.append(tuple(chance.choices(alphabet, k=chance.randint(1, 8))))
    gap_answer = sorted(chance.sample(range(len(answer)), chance.randint(1, len(answer))))
    return list(range(len(reference))), gap_answer, reference, answer, frozenset(reference)


@pytest.mark.parametrize("seed", range(300))
def test_the_large_gap_decides_as_the_small_one(seed: int) -> None:
    gap = _random_gap(seed)
    assert LargeGapResolver().resolve(*gap) == SmallGapResolver().resolve(*gap)
