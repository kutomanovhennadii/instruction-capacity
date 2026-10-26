"""Shared deterministic fixtures of the public test suite: a mock model without network."""

import pytest

from iec.contracts import GenParams, ModelConfig
from iec.model_client import ModelBuilder, ModelClient, Tokenizer, build_default_model_builder


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Return the minimal configuration of the registered mock provider."""
    return ModelConfig(model_id="mock-model", provider="mock", gen_params=GenParams(seed=42))


@pytest.fixture
def model_builder() -> ModelBuilder:
    """Return the default builder registry (mock and OpenAI providers)."""
    return build_default_model_builder()


@pytest.fixture
def mock_provider(
    model_builder: ModelBuilder, mock_model_config: ModelConfig
) -> tuple[ModelClient, Tokenizer]:
    """Build the mock model and tokenizer through the public builder path."""
    return model_builder.build(mock_model_config)
