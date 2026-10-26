"""Mirror tests of the contracts and of the mock provider."""

import logging
from collections.abc import Iterator

import pytest

from iec.contracts import GenParams, ModelConfig, Prompt, RawOutput, TokenUsage
from iec.infrastructure.logging import PACKAGE_LOGGER_NAME
from iec.model_client import (
    MockModelClient,
    MockTokenizer,
    ModelClient,
    Tokenizer,
    build_mock_provider,
)


@pytest.fixture(autouse=True)
def isolated_package_logger() -> Iterator[None]:
    """Do not leave global log handlers between tests."""
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers.clear()
    try:
        yield
    finally:
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def test_generate_is_deterministic_and_returns_raw_output() -> None:
    client = MockModelClient(model_id="mock-model")
    prompt = Prompt(text="lake cloud stone")
    params = GenParams(seed=42)

    first = client.generate(prompt, params)
    second = client.generate(prompt, params)

    assert isinstance(first, RawOutput)
    assert first == second
    assert first.text == prompt.text
    # The adapter returns the counter already in project fields, not in provider names.
    assert first.usage == TokenUsage(prompt_tokens=3, response_tokens=3)
    assert first.finish_reason == "stop"
    assert first.latency_s == 0.0


def test_generate_emits_model_telemetry(capsys: pytest.CaptureFixture[str]) -> None:
    client = MockModelClient(model_id="mock-model")

    client.generate(Prompt(text="lake"), GenParams())

    captured = capsys.readouterr()
    assert "[module=mock_provider step=model_generate model=mock-model]" in captured.err
    assert "step status=success" in captured.err


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("lake", True),
        (" lake", True),  # a leading separator space is the working probe form of the vocabulary
        ("two words", False),
        (" two words", False),
        ("", False),
        (" ", False),
        ("extraordinarilylong", False),
        (" extraordinarilylong", False),  # the space must not "buy back" the length
    ],
)
def test_is_single_token_uses_deterministic_rule(word: str, expected: bool) -> None:
    assert MockTokenizer().is_single_token(word) is expected


def test_count_tokens_counts_whitespace_separated_items() -> None:
    tokenizer = MockTokenizer()

    assert tokenizer.count_tokens("lake\ncloud  stone") == 3
    assert tokenizer.count_tokens("") == 0


def test_factory_uses_config_and_satisfies_protocols() -> None:
    config = ModelConfig(
        model_id="mock-model",
        provider="mock",
        gen_params=GenParams(),
        tokenizer_name="mock-v1",
    )

    client, tokenizer = build_mock_provider(config)

    assert isinstance(client, ModelClient)
    assert isinstance(tokenizer, Tokenizer)
    assert client.model_id == "mock-model"
    assert tokenizer.name == "mock-v1"
