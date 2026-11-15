"""Shared fixtures of the public test suite: isolation of the package logger and a mock model."""

import logging
from collections.abc import Iterator

import pytest

from iec.contracts import GenParams, ModelConfig
from iec.infrastructure.logging import PACKAGE_LOGGER_NAME
from iec.model_client import ModelBuilder, ModelClient, Tokenizer, build_default_model_builder


@pytest.fixture
def isolated_package_logger() -> Iterator[logging.Logger]:
    """Give a test the package logger without handlers and restore its state afterwards.

    The handlers a test adds are closed at the end: an open file handler would keep its file
    locked on Windows.
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    previous_handlers = logger.handlers[:]
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers.clear()
    try:
        yield logger
    finally:
        for handler in logger.handlers:
            if handler not in previous_handlers:
                handler.close()
        logger.handlers[:] = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


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
