"""Mirror tests of the word stream generator."""

from collections import Counter

import pytest

from iec.contracts import PointConditions, Vocab
from iec.infrastructure.exceptions import ConfigError
from iec.model_client import MockTokenizer
from iec.prompt_generator import RuleGenerator, TextGenerator
from iec.prompt_generator.text_generator import WORDS_PER_LINE

# Twenty words: enough for a lexicon with background and for the "vocab too small" check.
_WORDS = frozenset(
    {
        "lake", "stone", "dog", "cloud", "tree", "fish", "sun", "cat",
        "moon", "rock", "bird", "sand", "wolf", "star", "leaf", "rain",
        "snow", "wind", "fire", "hill",
    }
)


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


def _make_params(
    *,
    D: int = 6,  # noqa: N803 - protocol axis, one vocabulary with ``m``
    n: int = 10,
    m: int = 2,
    seed_text: int = 42,
) -> PointConditions:
    """Build the task conditions (by default D=6, n=10 -> v=60, m=2)."""
    return PointConditions(
        D=D,
        n=n,
        m=m,
        seed_rules=7,
        seed_text=seed_text,
    )


def _generate(params: PointConditions, vocab: Vocab | None = None) -> tuple:
    """Run the chain rules -> text on the mock tokenizer."""
    vocab = vocab if vocab is not None else _make_vocab()
    table = RuleGenerator(vocab).generate(params)
    text = TextGenerator(vocab, MockTokenizer()).generate(params, table)
    return table, text


def test_every_word_occurs_exactly_n_times() -> None:
    params = _make_params()

    _, text = _generate(params)

    counts = Counter(text.words)
    assert len(text.words) == params.v
    assert set(counts.values()) == {params.n}


def test_text_has_exactly_d_distinct_words_and_no_targets() -> None:
    params = _make_params()

    table, text = _generate(params)

    distinct = set(text.words)
    assert len(distinct) == params.D
    assert set(table.sources) <= distinct
    assert distinct.isdisjoint(table.targets)
    assert distinct <= _WORDS


def test_replacements_are_marked_in_reading_order() -> None:
    params = _make_params()

    table, text = _generate(params)

    assert text.replacement_count == table.m * params.n
    # The indices point exactly at the table words and cover all their occurrences.
    source_set = set(table.sources)
    marked = [text.words[index] for index in text.replacement_positions]
    assert set(marked) == source_set
    assert text.replacement_positions == tuple(
        index for index, word in enumerate(text.words) if word in source_set
    )


def test_len_tokens_is_measured_and_predictable_on_mock() -> None:
    params = _make_params()

    _, text = _generate(params)

    # The mock tokenizer counts elements via split() -> exactly v tokens.
    assert text.len_tokens == params.v
    assert text.tokenizer_name == "mock"


def test_lines_are_assembled_by_protocol_width() -> None:
    params = _make_params()

    _, text = _generate(params)

    lines = text.text.split("\n")
    assert text.words_per_line == WORDS_PER_LINE
    assert all(len(line.split()) == WORDS_PER_LINE for line in lines[:-1])
    assert 0 < len(lines[-1].split()) <= WORDS_PER_LINE


def test_same_seed_gives_identical_text() -> None:
    params = _make_params()

    _, first = _generate(params)
    _, second = _generate(params)

    assert first == second


def test_different_seed_gives_different_text() -> None:
    _, first = _generate(_make_params(seed_text=42))
    _, second = _generate(_make_params(seed_text=43))

    assert first.words != second.words


def test_lexicon_without_background_is_allowed() -> None:
    # Degenerate case D = m: the text consists of targets only.
    params = _make_params(D=2, n=10, m=2)

    table, text = _generate(params)

    assert set(text.words) == set(table.sources)


def test_text_length_is_divisible_by_construction() -> None:
    # The former check that v is a multiple of D was removed together with the length field: the
    # length is derived as D*n, so the inconsistency once caught by
    # ``TEXT_TOTAL_NOT_DIVISIBLE`` is impossible by construction of the contract.
    params = _make_params(D=6, n=7)

    _, text = _generate(params)

    assert len(text.words) == params.v
    assert params.v % params.D == 0


def test_distinct_below_m_raises_config_error() -> None:
    params = _make_params(D=2, n=30, m=3)
    vocab = _make_vocab()
    table = RuleGenerator(vocab).generate(params)

    with pytest.raises(ConfigError) as excinfo:
        TextGenerator(vocab, MockTokenizer()).generate(params, table)

    assert excinfo.value.error_code == "TEXT_DISTINCT_INVALID"


def test_vocab_too_small_for_lexicon_raises_config_error() -> None:
    # Vocabulary of 20 words: m=5 rules pass (10 words), but D=16 needs 16+5 > 20.
    params = _make_params(D=16, n=4, m=5)
    vocab = _make_vocab()
    table = RuleGenerator(vocab).generate(params)

    with pytest.raises(ConfigError) as excinfo:
        TextGenerator(vocab, MockTokenizer()).generate(params, table)

    assert excinfo.value.error_code == "TEXT_DISTINCT_INVALID"
