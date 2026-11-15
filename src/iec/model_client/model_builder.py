"""The model builder and the single registry of provider factories."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from iec.contracts import ModelConfig
from iec.infrastructure.exceptions import ConfigError, IECError, ProviderError
from iec.infrastructure.telemetry import TelemetryHub
from iec.model_client.model_client import ModelClient
from iec.model_client.provider_adapters.mock_provider import build_mock_provider
from iec.model_client.provider_adapters.openai_client import (
    OpenAIModelClient,
    build_openai_http_client,
)
from iec.model_client.tokenizer import Tokenizer
from iec.model_client.tokenizer_adapters.tiktoken_tokenizer import (
    TiktokenTokenizer,
    build_tiktoken_tokenizer,
)

ProviderFactory = Callable[[ModelConfig], tuple[ModelClient, Tokenizer]]

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass(slots=True)
class ModelBuilder:
    """Assembles the pair of model and tokenizer by the provider key from the config.

    The registry is the only source of truth about available providers. After assembly the
    bench works only with ``ModelClient`` and ``Tokenizer`` and does not interpret
    ``model_id``.
    """

    factories: dict[str, ProviderFactory] = field(default_factory=dict)

    def known_providers(self) -> frozenset[str]:
        """Return the immutable set of registered provider keys."""
        return frozenset(self.factories)

    def build(self, config: ModelConfig) -> tuple[ModelClient, Tokenizer]:
        """Assemble the model and the tokenizer from the configuration.

        Args:
            config: Configuration with the key of a registered provider.

        Returns:
            A pair of objects behind the universal project contracts.

        Raises:
            ConfigError: The provider is not registered.
            ProviderError: ``PROVIDER_BUILD_FAILED`` - the factory raised a project error, such as
                a missing key or a tokenizer that does not load; the original error is kept in
                ``__cause__``. Any other exception of a factory is a defect and passes unchanged.

        Side effects:
            The factory of a live provider loads the tokenizer and opens a connection pool
            (provider ``"openai"``); the mock provider has no side effects.
        """
        factory = self.factories.get(config.provider)
        if factory is None:
            raise ConfigError(
                f"unknown model provider '{config.provider}' (known: {sorted(self.factories)})",
                error_code="UNKNOWN_PROVIDER",
            )
        try:
            return factory(config)
        except IECError as cause:
            raise ProviderError.from_cause(
                f"failed to build model '{config.model_id}' with provider '{config.provider}'",
                cause,
                error_code="PROVIDER_BUILD_FAILED",
            ) from cause


def build_openai_provider(
    config: ModelConfig, telemetry: TelemetryHub | None = None
) -> tuple[OpenAIModelClient, TiktokenTokenizer]:
    """Assemble the live OpenAI pair: the Chat Completions adapter and the tiktoken tokenizer.

    The API key is read from the environment and checked here, before the first model call:
    a series without a key must not start in the middle of the matrix. The HTTP client is
    assembled once per series and lives as a field of the adapter - keep-alive reuses the
    connection between points.

    Args:
        config: Configuration with the ``model_id`` of the model snapshot and the tokenizer
            name.
        telemetry: Hub that records the calls of the model; without it the model writes the log
            line of every call only.

    Returns:
        A pair of concrete types; outside the layer it goes behind the project contracts.

    Raises:
        ConfigError: ``OPENAI_API_KEY_MISSING`` - the environment variable is not set.
        ProviderError: ``TIKTOKEN_LOAD_FAILED`` - tiktoken does not know such a model.

    Side effects:
        Opens an httpx connection pool (without network calls before the first
        ``generate``) and loads the tiktoken encoder from the package.
    """
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ConfigError(
            f"{OPENAI_API_KEY_ENV} is not set — "
            "see the README of iec.model_client, section 'API key'",
            error_code="OPENAI_API_KEY_MISSING",
        )
    # The tokenizer name is part of the vocabulary identification, so it is set explicitly in
    # the series config; tiktoken would also parse the snapshot name of the model, but
    # matching the specification of the intersection is more important.
    tokenizer = build_tiktoken_tokenizer(config.tokenizer_name or config.model_id)
    http_client = build_openai_http_client(api_key)
    hub = telemetry if telemetry is not None else TelemetryHub.log_only()
    client = OpenAIModelClient(model_id=config.model_id, client=http_client, telemetry=hub)
    return client, tokenizer


def build_default_model_builder(telemetry: TelemetryHub | None = None) -> ModelBuilder:
    """Return the standard registry: the deterministic mock and the live OpenAI provider.

    Args:
        telemetry: Hub handed to every model the builder assembles; without it the models write
            the log line of every call only.
    """
    hub = telemetry if telemetry is not None else TelemetryHub.log_only()

    def build_mock(config: ModelConfig) -> tuple[ModelClient, Tokenizer]:
        """Build the mock pair observed by the hub of the builder."""
        return build_mock_provider(config, hub)

    def build_openai(config: ModelConfig) -> tuple[ModelClient, Tokenizer]:
        """Build the OpenAI pair observed by the hub of the builder."""
        return build_openai_provider(config, hub)

    return ModelBuilder(factories={"mock": build_mock, "openai": build_openai})
