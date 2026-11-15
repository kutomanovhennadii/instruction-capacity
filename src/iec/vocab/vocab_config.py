"""Pydantic configuration of building the vocabulary ``V`` and its loading from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from iec.infrastructure.configuration import load_yaml_model


class TokenizerSpec(BaseModel):
    """Specification of one tokenizer for the single-token check.

    ``provider`` is the key of the tokenizer registry of ``iec.model_client`` (``"tiktoken"`` |
    ``"hf"`` | ``"hfjson"`` | ``"gemini_count"``); ``name`` is the model name or path by which
    the tokenizer is loaded (``"gpt-4o"``, ``"Qwen/Qwen2.5-0.5B"`` and the like).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str


class VocabBuildConfig(BaseModel):
    """Declarative parameters of building the vocabulary ``V``.

    Loaded from YAML by :func:`load_vocab_config`; all filtering parameters live here, not
    in the code. ``tokenizer_specs`` sets the tokenizer set over which the intersection of
    single-token sets is taken. ``format_version`` is the version of the file format
    (specification §5): version 1 is the only one, and a file without the field is of version 1.
    """

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    vocab_id: str
    source: str = "cmudict"
    max_syllables: int = 1
    ascii_lower_only: bool = True
    tokenizer_specs: list[TokenizerSpec]

    @field_validator("tokenizer_specs")
    @classmethod
    def _require_at_least_one_tokenizer(cls, specs: list[TokenizerSpec]) -> list[TokenizerSpec]:
        """Forbid an empty set: the intersection without tokenizers is undefined."""
        if not specs:
            raise ValueError("tokenizer_specs must list at least one tokenizer")
        return specs

    @field_validator("max_syllables")
    @classmethod
    def _require_positive_syllables(cls, value: int) -> int:
        """The number of syllables is positive (0 syllables would cut off every word)."""
        if value < 1:
            raise ValueError("max_syllables must be >= 1")
        return value


def load_vocab_config(path: Path) -> VocabBuildConfig:
    """Load and validate the vocabulary build configuration from a YAML file.

    Args:
        path: Path to the YAML config (for example ``config/vocab.yaml``).

    Returns:
        The validated configuration.

    Raises:
        ConfigError: the file is not found, the YAML is invalid or the schema is violated
            (``error_code="VOCAB_CONFIG_INVALID"``); the original cause is kept.
    """
    return load_yaml_model(
        path, VocabBuildConfig, error_code="VOCAB_CONFIG_INVALID", description="vocab config"
    )
