"""Shared fixtures of the measuring layer: a real task from the task generation chain."""

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from iec.contracts import PointConditions, RuleTable, SourceText, Vocab
from iec.model_client import MockTokenizer
from iec.prompt_generator import RuleGenerator, TextGenerator

# Twenty words - enough for a lexicon with background at D + m <= size.
# fmt: off
_WORDS = frozenset(
    {
        "lake", "stone", "dog", "cloud", "tree", "fish", "sun", "cat",
        "moon", "rock", "bird", "sand", "wolf", "star", "leaf", "rain",
        "snow", "wind", "fire", "hill",
    }
)
# fmt: on


@dataclass(frozen=True, slots=True)
class Task:
    """A generated task: conditions, the replacement table and the word stream."""

    params: PointConditions
    rule_table: RuleTable
    source_text: SourceText


TaskFactory = Callable[..., Task]


@pytest.fixture
def vocab() -> Vocab:
    """Return a test vocabulary with the canonical hash."""
    words = _WORDS
    return Vocab(
        vocab_id="v_test",
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("mock",),
        source="test",
        build_params={},
    )


@pytest.fixture
def make_task(vocab: Vocab) -> TaskFactory:
    """Return a task factory by the generation chain (by default D=6, n=10 -> v=60, 20 trials)."""

    def factory(
        *,
        D: int = 6,  # noqa: N803 - protocol axis, one vocabulary with ``m``
        n: int = 10,
        m: int = 2,
        seed_rules: int = 7,
        seed_text: int = 42,
    ) -> Task:
        params = PointConditions(
            D=D,
            n=n,
            m=m,
            seed_rules=seed_rules,
            seed_text=seed_text,
        )
        rule_table = RuleGenerator(vocab).generate(params)
        source_text = TextGenerator(vocab, MockTokenizer()).generate(params, rule_table)
        return Task(params=params, rule_table=rule_table, source_text=source_text)

    return factory


@pytest.fixture
def task(make_task: TaskFactory) -> Task:
    """Return the standard task for the tests of the measuring layer."""
    return make_task()
