"""Mirror tests of the model builder and the provider registry."""

import pytest

from iec.contracts import GenParams, ModelConfig
from iec.infrastructure.exceptions import ConfigError, ProviderError
from iec.model_client import (
    MockModelClient,
    MockTokenizer,
    ModelBuilder,
    ModelClient,
    OpenAIModelClient,
    TiktokenTokenizer,
    Tokenizer,
    build_default_model_builder,
    build_openai_provider,
)
from iec.model_client.model_builder import OPENAI_API_KEY_ENV

PILOT_MODEL_ID = "gpt-4o-mini-2024-07-18"
PILOT_TOKENIZER_NAME = "gpt-4o-mini"


def _openai_config() -> ModelConfig:
    """Return the build configuration of the live provider of the pilot."""
    return ModelConfig(
        model_id=PILOT_MODEL_ID,
        provider="openai",
        gen_params=GenParams(),
        tokenizer_name=PILOT_TOKENIZER_NAME,
    )


def test_default_builder_builds_mock_model(
    model_builder: ModelBuilder,
    mock_model_config: ModelConfig,
) -> None:
    client, tokenizer = model_builder.build(mock_model_config)

    assert isinstance(client, MockModelClient)
    assert isinstance(tokenizer, MockTokenizer)
    assert isinstance(client, ModelClient)
    assert isinstance(tokenizer, Tokenizer)
    assert client.model_id == mock_model_config.model_id


def test_default_builder_knows_mock_and_openai_providers() -> None:
    builder = build_default_model_builder()

    # Connecting a model = an adapter + an entry here.
    assert builder.known_providers() == frozenset({"mock", "openai"})


def test_unknown_provider_is_configuration_error() -> None:
    builder = build_default_model_builder()
    config = ModelConfig(
        model_id="unknown-model",
        provider="missing",
        gen_params=GenParams(),
    )

    with pytest.raises(ConfigError) as caught:
        builder.build(config)

    assert caught.value.error_code == "UNKNOWN_PROVIDER"
    assert "missing" in str(caught.value)
    assert "mock" in str(caught.value)


def test_factory_failure_is_wrapped_with_original_cause() -> None:
    cause = RuntimeError("adapter initialization failed")

    def failing_factory(config: ModelConfig) -> tuple[ModelClient, Tokenizer]:
        del config
        raise cause

    builder = ModelBuilder(factories={"broken": failing_factory})
    config = ModelConfig(
        model_id="broken-model",
        provider="broken",
        gen_params=GenParams(),
    )

    with pytest.raises(ProviderError) as caught:
        builder.build(config)

    assert caught.value.error_code == "PROVIDER_BUILD_FAILED"
    assert caught.value.__cause__ is cause


def test_mock_provider_fixture_uses_public_builder_path(
    mock_provider: tuple[ModelClient, Tokenizer],
) -> None:
    client, tokenizer = mock_provider

    assert isinstance(client, ModelClient)
    assert isinstance(tokenizer, Tokenizer)


def test_openai_factory_without_api_key_is_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)

    with pytest.raises(ConfigError) as caught:
        build_openai_provider(_openai_config())

    # Checked before the first call: a series without a key must not start in the middle.
    assert caught.value.error_code == "OPENAI_API_KEY_MISSING"
    assert OPENAI_API_KEY_ENV in str(caught.value)


def test_openai_factory_builds_live_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "sk-test-key")

    client, tokenizer = build_openai_provider(_openai_config())
    try:
        assert isinstance(client, OpenAIModelClient)
        assert isinstance(tokenizer, TiktokenTokenizer)
        assert isinstance(client, ModelClient)
        assert isinstance(tokenizer, Tokenizer)
        assert client.model_id == PILOT_MODEL_ID
        assert tokenizer.name == f"tiktoken/{PILOT_TOKENIZER_NAME}"
    finally:
        client.close()


def test_default_builder_builds_openai_pair_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV, "sk-test-key")
    builder = build_default_model_builder()

    client, tokenizer = builder.build(_openai_config())
    try:
        assert isinstance(client, OpenAIModelClient)
        assert isinstance(tokenizer, TiktokenTokenizer)
    finally:
        client.close()


def test_openai_factory_missing_key_surfaces_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_API_KEY_ENV, raising=False)
    builder = build_default_model_builder()

    # The builder normalizes the factory failure into ProviderError, keeping the cause.
    with pytest.raises(ProviderError) as caught:
        builder.build(_openai_config())

    assert caught.value.error_code == "PROVIDER_BUILD_FAILED"
    assert isinstance(caught.value.__cause__, ConfigError)
    assert caught.value.__cause__.error_code == "OPENAI_API_KEY_MISSING"
