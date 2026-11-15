"""The tokenizer builder and the single registry of tokenizer factories.

A tokenizer is chosen by its provider - the library or the service that tokenizes - and by the
name of the model inside it: ``tiktoken`` and ``gpt-4o-mini``, ``hfjson`` and a repository of the
Hugging Face Hub. The registry maps a provider to its factory, so a new tokenizer is a new adapter
and a new entry here (specification §2.10). The factories of the two Hugging Face providers import
their adapters only when they are called: importing the package loads neither ``transformers``
nor ``tokenizers``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from iec.infrastructure.exceptions import ConfigError
from iec.model_client.tokenizer import Tokenizer
from iec.model_client.tokenizer_adapters.gemini_count_tokenizer import (
    GEMINI_API_KEY_ENV,
    GEMINI_COUNT_PROVIDER,
    build_gemini_count_client,
    build_gemini_count_tokenizer,
)
from iec.model_client.tokenizer_adapters.tiktoken_tokenizer import (
    TIKTOKEN_PROVIDER,
    build_tiktoken_tokenizer,
)

#: Factory of one provider: takes the name of the model inside the provider, returns the tokenizer.
TokenizerFactory = Callable[[str], Tokenizer]

#: Keys of the two Hugging Face providers. They equal ``HF_PROVIDER`` and ``HUB_PROVIDER`` of the
#: adapters and are written here as values, because importing the adapters would load their
#: libraries; the mirror test checks the equality.
HF_PROVIDER_KEY = "hf"
HUB_PROVIDER_KEY = "hfjson"


@dataclass(slots=True)
class TokenizerBuilder:
    """Builds a tokenizer by its provider and model name from the registry of factories.

    The registry is the only source of truth about the available tokenizers; outside the model
    layer a tokenizer is seen only through the ``Tokenizer`` contract.
    """

    factories: dict[str, TokenizerFactory] = field(default_factory=dict)

    def known_providers(self) -> frozenset[str]:
        """Return the immutable set of registered provider keys."""
        return frozenset(self.factories)

    def build(self, provider: str, name: str) -> Tokenizer:
        """Build the tokenizer of the model ``name`` of the provider.

        Args:
            provider: Key of a registered provider, for example ``"tiktoken"``.
            name: Name of the model inside the provider, for example ``"gpt-4o-mini"``.

        Returns:
            The tokenizer behind the project contract.

        Raises:
            ConfigError: ``UNKNOWN_TOKENIZER_PROVIDER`` - the provider is not registered; a
                factory may raise its own, such as ``GEMINI_API_KEY_MISSING``.
            ProviderError: the factory could not load the tokenizer; the cause is kept.

        Side effects:
            A Hugging Face factory downloads the tokenizer files on the first load; the count of
            a Gemini model opens an HTTP client.
        """
        factory = self.factories.get(provider)
        if factory is None:
            raise ConfigError(
                f"unknown tokenizer provider '{provider}' (known: {sorted(self.factories)})",
                error_code="UNKNOWN_TOKENIZER_PROVIDER",
            )
        return factory(name)


def build_default_tokenizer_builder() -> TokenizerBuilder:
    """Return the standard registry: tiktoken, two Hugging Face providers and the Gemini count."""
    return TokenizerBuilder(
        factories={
            TIKTOKEN_PROVIDER: build_tiktoken_tokenizer,
            HF_PROVIDER_KEY: _build_hf_tokenizer,
            HUB_PROVIDER_KEY: _build_hub_tokenizer,
            GEMINI_COUNT_PROVIDER: _build_gemini_count_tokenizer,
        }
    )


def _build_hf_tokenizer(model_name: str) -> Tokenizer:
    """Build a Hugging Face tokenizer, importing its adapter only now: it loads transformers."""
    from iec.model_client.tokenizer_adapters.hf_tokenizer import build_hf_tokenizer

    return build_hf_tokenizer(model_name)


def _build_hub_tokenizer(repo_id: str) -> Tokenizer:
    """Build a tokenizer from a file, importing its adapter only now: it loads tokenizers."""
    from iec.model_client.tokenizer_adapters.hub_tokenizer import build_hub_tokenizer

    return build_hub_tokenizer(repo_id)


def _build_gemini_count_tokenizer(model_id: str) -> Tokenizer:
    """Build the count of a Gemini model with the key from ``GEMINI_API_KEY``.

    Raises:
        ConfigError: ``GEMINI_API_KEY_MISSING`` - the variable is not set.

    Side effects:
        Opens an HTTP client of the Gemini API; there are no network calls before the first count.
    """
    key = os.environ.get(GEMINI_API_KEY_ENV)
    if not key:
        raise ConfigError(
            f"{GEMINI_API_KEY_ENV} is not set: the tokenizer {GEMINI_COUNT_PROVIDER} needs it",
            error_code="GEMINI_API_KEY_MISSING",
        )
    return build_gemini_count_tokenizer(model_id, build_gemini_count_client(key))
