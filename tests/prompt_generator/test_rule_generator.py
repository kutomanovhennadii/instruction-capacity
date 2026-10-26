"""Mirror tests of the replacement table generator."""

import pytest

from iec.contracts import PointConditions, Vocab
from iec.infrastructure.exceptions import ConfigError, VocabError
from iec.prompt_generator import RuleGenerator
from iec.prompt_generator.rule_generator import _telemetry_context

# Twenty words -> at most ten pairs; enough both for success and for "vocab too small".
_WORDS = frozenset(
    {
        "lake", "stone", "dog", "cloud", "tree", "fish", "sun", "cat",
        "moon", "rock", "bird", "sand", "wolf", "star", "leaf", "rain",
        "snow", "wind", "fire", "hill",
    }
)

_STRIDE = 1_000_003  # legacy permutation stride: tests prove the public generator repeats it


def _make_vocab(words: frozenset[str] = _WORDS) -> Vocab:
    """Build a test vocabulary with the canonical hash."""
    return Vocab(
        vocab_id="v_test",
        words=words,
        hash=Vocab.compute_hash(words),
        tokenizer_names=("mock",),
        source="test",
        build_params={},
    )


def _make_params(m: int, *, seed_rules: int = 7, q: int = 1) -> PointConditions:
    """Build point conditions; the order seed reproduces the frozen generator for this q."""
    return PointConditions(
        D=10,
        n=10,
        m=m,
        seed_rules=seed_rules,
        seed_text=42,
        seed_order=0 if q == 1 else seed_rules * _STRIDE + q,
    )


def test_table_is_bijective_disjoint_and_from_vocab() -> None:
    vocab = _make_vocab()
    table = RuleGenerator(vocab).generate(_make_params(m=5))

    # Each word to replace has exactly one replacement, all 2m words are distinct.
    assert table.m == 5
    all_words = table.sources + table.targets
    assert len(set(all_words)) == 10
    # The groups do not intersect, all words come from the vocabulary.
    assert set(table.sources).isdisjoint(table.targets)
    assert set(all_words) <= vocab.words


def test_table_records_vocab_id_and_seed() -> None:
    table = RuleGenerator(_make_vocab()).generate(_make_params(m=3, seed_rules=99))

    assert table.vocab_id == "v_test"
    assert table.seed_rules == 99


def test_same_seed_gives_identical_table_including_order() -> None:
    params = _make_params(m=5, seed_rules=7)

    first = RuleGenerator(_make_vocab()).generate(params)
    second = RuleGenerator(_make_vocab()).generate(params)

    assert first == second


def test_different_seed_gives_different_table() -> None:
    generator = RuleGenerator(_make_vocab())

    first = generator.generate(_make_params(m=5, seed_rules=7))
    second = generator.generate(_make_params(m=5, seed_rules=8))

    assert first != second


def test_max_pairs_is_reachable() -> None:
    vocab = _make_vocab()
    table = RuleGenerator(vocab).generate(_make_params(m=vocab.max_pairs))

    assert table.m == vocab.max_pairs


def test_explicit_q_one_equals_default_order() -> None:
    """q = 1 is the drawn order: no permutation is applied at all."""
    generator = RuleGenerator(_make_vocab())

    default = generator.generate(_make_params(m=8))
    explicit = generator.generate(_make_params(m=8, q=1))

    assert default == explicit


def test_permutation_keeps_pairs_and_changes_order() -> None:
    """The same pairs in a different order - this separates c_p from the difficulty of pairs."""
    generator = RuleGenerator(_make_vocab())

    first = generator.generate(_make_params(m=8, q=1))
    permuted = generator.generate(_make_params(m=8, q=2))

    assert set(zip(permuted.sources, permuted.targets, strict=True)) == set(
        zip(first.sources, first.targets, strict=True)
    )
    assert permuted.rules != first.rules


def test_different_q_gives_different_orders() -> None:
    generator = RuleGenerator(_make_vocab())

    orders = {generator.generate(_make_params(m=8, q=q)).sources for q in (1, 2, 3, 4)}

    assert len(orders) == 4


def test_same_seed_and_q_give_identical_table() -> None:
    """Determinism from the pair (seed_rules, q): the table is reproduced to the letter."""
    params = _make_params(m=8, seed_rules=99, q=3)

    first = RuleGenerator(_make_vocab()).generate(params)
    second = RuleGenerator(_make_vocab()).generate(params)

    assert first == second


def test_permutation_depends_on_seed_as_well_as_q() -> None:
    """The seed and the permutation number do not merge: different seeds at the same q differ."""
    generator = RuleGenerator(_make_vocab())

    first = generator.generate(_make_params(m=8, seed_rules=7, q=2))
    second = generator.generate(_make_params(m=8, seed_rules=8, q=2))

    assert first != second


def test_single_rule_table_is_permutation_invariant() -> None:
    """m = 1: every permutation is the identity - not an error but a degenerate case."""
    generator = RuleGenerator(_make_vocab())

    first = generator.generate(_make_params(m=1, q=1))
    permuted = generator.generate(_make_params(m=1, q=5))

    assert permuted == first


def test_negative_order_seed_raises_config_error() -> None:
    generator = RuleGenerator(_make_vocab())
    conditions = PointConditions(D=10, n=10, m=5, seed_rules=7, seed_text=42, seed_order=-1)

    with pytest.raises(ConfigError) as excinfo:
        generator.generate(conditions)

    assert excinfo.value.error_code == "RULES_SEED_ORDER_INVALID"


def test_telemetry_context_carries_the_permutation_axis() -> None:
    """The point key in the event is incomplete without the order: permuted points merge."""
    params = _make_params(m=5, seed_rules=99, q=4)

    context = _telemetry_context(RuleGenerator(_make_vocab()), params)

    assert (context.m, context.seed_rules, context.seed_order) == (5, 99, 99 * _STRIDE + 4)


def test_m_below_one_raises_config_error() -> None:
    generator = RuleGenerator(_make_vocab())

    with pytest.raises(ConfigError) as excinfo:
        generator.generate(_make_params(m=0))

    assert excinfo.value.error_code == "RULES_M_INVALID"


def test_vocab_too_small_raises_config_error() -> None:
    vocab = _make_vocab()

    with pytest.raises(ConfigError) as excinfo:
        RuleGenerator(vocab).generate(_make_params(m=vocab.max_pairs + 1))

    assert excinfo.value.error_code == "RULES_VOCAB_TOO_SMALL"


def test_empty_vocab_raises_vocab_error_on_init() -> None:
    with pytest.raises(VocabError) as excinfo:
        RuleGenerator(_make_vocab(frozenset()))

    assert excinfo.value.error_code == "RULES_EMPTY_VOCAB"
