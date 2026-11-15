"""Mirror tests of the point executor: statuses, limits of the model, telemetry, records."""

from pathlib import Path

import pytest

from iec.contracts import (
    GenParams,
    ModelConfig,
    PointConditions,
    PointStatus,
    Prompt,
    RawOutput,
    Vocab,
)
from iec.infrastructure.exceptions import ProviderError
from iec.infrastructure.telemetry import (
    POINT_EXECUTION,
    MemorySink,
    ObservationLevels,
    TelemetryHub,
)
from iec.model_client import ModelClient, Tokenizer
from iec.point_execution import (
    EMPTY_ANSWER_CODE,
    PROVIDER_TIMEOUT_CODE,
    WRAPPER_RESERVE_TOKENS,
    PointExecutor,
    append_record,
    ceiling_verdict,
    read_records,
    task_from_record,
)
from iec.vocab import VocabStore

_VOCAB_DIR = Path(__file__).resolve().parents[2] / "assets" / "vocab"
_CONDITIONS = PointConditions(D=16, n=5, m=8, seed_rules=11, seed_text=12, seed_order=0)


@pytest.fixture
def vocab() -> Vocab:
    return VocabStore(_VOCAB_DIR).load("v1_intersection")


def _config(
    max_tokens: int = 4096,
    context_window_tokens: int | None = None,
    response_ceiling_tokens: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        model_id="mock-model",
        provider="mock",
        gen_params=GenParams(max_tokens=max_tokens),
        context_window_tokens=context_window_tokens,
        response_ceiling_tokens=response_ceiling_tokens,
    )


class _FailingClient:
    def __init__(self, code: str) -> None:
        self._code = code

    @property
    def model_id(self) -> str:
        return "failing"

    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        raise ProviderError("provider refused", error_code=self._code)


def test_mock_echo_is_an_ok_point_with_an_answer(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    client, tokenizer = mock_provider
    outcome = PointExecutor(client, tokenizer, vocab, _config()).execute(_CONDITIONS)

    assert outcome.record.status is PointStatus.OK
    assert outcome.raw_output is not None
    assert outcome.record.source_text == outcome.source_text.text
    assert outcome.record.rules == tuple((r.source, r.target) for r in outcome.rule_table.rules)
    assert outcome.record.answer == outcome.prompt.text


def test_point_beyond_the_window_is_na_without_a_call(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    client, tokenizer = mock_provider
    outcome = PointExecutor(
        client, tokenizer, vocab, _config(context_window_tokens=10)
    ).execute(_CONDITIONS)

    assert outcome.record.status is PointStatus.NA
    assert outcome.record.answer is None
    assert outcome.raw_output is None


def test_provider_failure_is_api_error_without_an_answer(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    _, tokenizer = mock_provider
    outcome = PointExecutor(
        _FailingClient("PROVIDER_API_ERROR"), tokenizer, vocab, _config()
    ).execute(_CONDITIONS)

    assert outcome.record.status is PointStatus.API_ERROR
    assert outcome.error_code == "PROVIDER_API_ERROR"
    assert outcome.raw_output is None


class _ScriptedClient:
    """A client that answers with the prepared texts in turn."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "scripted"

    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        self.calls += 1
        return RawOutput(text=self._texts[self.calls - 1], finish_reason="stop")


def test_silence_until_the_doubled_limit_is_api_error_without_an_answer(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    _, tokenizer = mock_provider
    outcome = PointExecutor(
        _FailingClient(PROVIDER_TIMEOUT_CODE), tokenizer, vocab, _config()
    ).execute(_CONDITIONS)

    assert outcome.record.status is PointStatus.API_ERROR
    assert outcome.record.answer is None
    assert outcome.error_code == PROVIDER_TIMEOUT_CODE


def test_an_empty_answer_is_asked_once_more(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    _, tokenizer = mock_provider
    client = _ScriptedClient("  \n", "lake cloud")
    outcome = PointExecutor(client, tokenizer, vocab, _config()).execute(_CONDITIONS)

    assert client.calls == 2
    assert outcome.record.status is PointStatus.OK
    assert outcome.record.answer == "lake cloud"


def test_a_second_empty_answer_is_api_error(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    _, tokenizer = mock_provider
    client = _ScriptedClient("", " ")
    outcome = PointExecutor(client, tokenizer, vocab, _config()).execute(_CONDITIONS)

    assert client.calls == 2
    assert outcome.record.status is PointStatus.API_ERROR
    assert outcome.record.answer is None
    assert outcome.error_code == EMPTY_ANSWER_CODE


def test_ceiling_verdict_requires_the_wrapper_reserve() -> None:
    config = _config(max_tokens=100)
    assert ceiling_verdict(config, 10, 100 - WRAPPER_RESERVE_TOKENS) is None
    assert ceiling_verdict(config, 10, 100 - WRAPPER_RESERVE_TOKENS + 1) is not None


def test_record_round_trip_and_task_rebuild(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab, tmp_path: Path
) -> None:
    client, tokenizer = mock_provider
    outcome = PointExecutor(client, tokenizer, vocab, _config()).execute(_CONDITIONS)
    path = tmp_path / "points.jsonl"
    append_record(path, outcome.record)
    append_record(path, outcome.record)

    records = read_records(path)
    table, text = task_from_record(records[0], vocab.vocab_id)

    assert records == [outcome.record, outcome.record]
    assert table == outcome.rule_table
    assert text.words == outcome.source_text.words
    assert text.replacement_positions == outcome.source_text.replacement_positions


def test_the_hub_observes_the_execution_and_the_generation_of_the_task(
    mock_provider: tuple[ModelClient, Tokenizer], vocab: Vocab
) -> None:
    client, tokenizer = mock_provider
    sink = MemorySink()
    executor = PointExecutor(
        client, tokenizer, vocab, _config(), telemetry=TelemetryHub(ObservationLevels(), [sink])
    )

    executor.execute(_CONDITIONS)

    operations = [event.operation for event in sink.events]
    assert operations == ["rules_generate", "text_generate", "point_execute"]
    event = sink.events[-1]
    context = event.context
    assert event.block == POINT_EXECUTION
    assert (context.model_id, context.m, context.seed_rules, context.seed_text) == (
        "mock-model",
        8,
        11,
        12,
    )
