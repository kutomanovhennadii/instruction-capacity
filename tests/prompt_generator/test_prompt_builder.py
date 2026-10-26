"""Mirror tests of the prompt builder and an end-to-end run of the generation chain."""

from iec.contracts import PointConditions, Rule, RuleTable, SourceText, Vocab
from iec.model_client import MockTokenizer
from iec.prompt_generator import PromptBuilder, RuleGenerator, TextGenerator

_WORDS = frozenset(
    {
        "lake", "stone", "dog", "cloud", "tree", "fish", "sun", "cat",
        "moon", "rock", "bird", "sand", "wolf", "star", "leaf", "rain",
        "snow", "wind", "fire", "hill",
    }
)


def _make_table() -> RuleTable:
    """Build a table of two rules in a meaningful order."""
    return RuleTable(
        rules=(Rule(source="lake", target="stone"), Rule(source="dog", target="cloud")),
        vocab_id="v_test",
        seed_rules=7,
    )


def _make_text() -> SourceText:
    """Build a small stream with one incomplete line."""
    return SourceText(
        words=("tree", "dog", "fish", "lake", "sun"),
        words_per_line=3,
        replacement_positions=(1, 3),
        len_tokens=5,
        tokenizer_name="mock",
        seed_text=42,
    )


def test_prompt_contains_rules_in_table_order() -> None:
    prompt = PromptBuilder().build(_make_table(), _make_text())

    first = prompt.text.index("lake -> stone")
    second = prompt.text.index("dog -> cloud")
    assert first < second


def test_prompt_contains_text_verbatim() -> None:
    text = _make_text()

    prompt = PromptBuilder().build(_make_table(), text)

    assert text.text in prompt.text
    assert "tree dog fish\nlake sun" in prompt.text


def test_payload_position_is_empty_on_this_milestone() -> None:
    prompt = PromptBuilder().build(_make_table(), _make_text())

    assert prompt.payload_position is None


def test_build_is_deterministic() -> None:
    builder = PromptBuilder()

    assert builder.build(_make_table(), _make_text()) == builder.build(_make_table(), _make_text())


def test_end_to_end_chain_composes_on_stubs() -> None:
    """End-to-end run: rules -> text -> prompt on a vocabulary and the mock tokenizer."""
    vocab = Vocab(
        vocab_id="v_test",
        words=_WORDS,
        hash=Vocab.compute_hash(_WORDS),
        tokenizer_names=("mock",),
        source="test",
        build_params={},
    )
    params = PointConditions(D=6, n=10, m=2, seed_rules=7, seed_text=42)
    params_snapshot = params.as_dict()

    table = RuleGenerator(vocab).generate(params)
    text = TextGenerator(vocab, MockTokenizer()).generate(params, table)
    prompt = PromptBuilder().build(table, text)

    # The contract values match the task conditions.
    assert table.m == params.m
    assert len(text.words) == params.v
    assert len(set(text.words)) == params.D
    # The conditions object is not changed and is fit to be passed further.
    assert params.as_dict() == params_snapshot
    # The prompt contains all rules and the text verbatim.
    for rule in table.rules:
        assert f"{rule.source} -> {rule.target}" in prompt.text
    assert text.text in prompt.text
    assert prompt.payload_position is None
