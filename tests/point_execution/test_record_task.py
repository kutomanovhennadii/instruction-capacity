"""Mirror tests of the rebuilding of a task from its published record."""

from iec.contracts import PointConditions, PointRecord, PointStatus
from iec.model_client import MockTokenizer
from iec.point_execution import task_from_record
from iec.prompt_generator.text_generator import WORDS_PER_LINE

_RECORD = PointRecord(
    conditions=PointConditions(D=3, n=2, m=1, seed_rules=4, seed_text=5, seed_order=0),
    rules=(("lake", "stone"),),
    source_text="lake tree dog\ntree lake dog",
    answer=None,
    status=PointStatus.NA,
    prompt_tokens=None,
    response_tokens=None,
)


def test_the_task_of_a_record_marks_the_left_words_of_the_rules() -> None:
    table, text = task_from_record(_RECORD, "v1_intersection")

    assert [(rule.source, rule.target) for rule in table.rules] == [("lake", "stone")]
    assert (table.vocab_id, table.seed_rules) == ("v1_intersection", 4)
    assert text.words == ("lake", "tree", "dog", "tree", "lake", "dog")
    assert text.replacement_positions == (0, 4)
    assert (text.words_per_line, text.seed_text) == (WORDS_PER_LINE, 5)
    assert (text.len_tokens, text.tokenizer_name) == (0, "unmeasured")


def test_a_given_tokenizer_measures_the_length_of_the_text() -> None:
    tokenizer = MockTokenizer()

    _, text = task_from_record(_RECORD, "v1_intersection", tokenizer)

    assert text.len_tokens == tokenizer.count_tokens(_RECORD.source_text)
    assert text.tokenizer_name == tokenizer.name
