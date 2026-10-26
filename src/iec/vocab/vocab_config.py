"""Pydantic configuration of building the vocabulary ``V`` and its loading from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from iec.infrastructure.exceptions import ConfigError


class TokenizerSpec(BaseModel):
    """Specification of one tokenizer for the single-token check.

    ``provider`` is the factory key (``"tiktoken"`` | ``"hf"``); ``name`` is the model
    name or path by which the tokenizer is loaded (``"gpt-4o"``, ``"Qwen/Qwen2.5-0.5B"`` and
    the like).
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    name: str


class VocabBuildConfig(BaseModel):
    """Declarative parameters of building the vocabulary ``V``.

    Loaded from YAML by :func:`load_vocab_config`; all filtering parameters live here, not
    in the code. ``tokenizer_specs`` sets the tokenizer set over which the intersection of
    single-token sets is taken.
    """

    model_config = ConfigDict(extra="forbid")

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
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as cause:
        raise ConfigError.from_cause(
            f"vocab config not found or unreadable: '{path}'",
            cause,
            error_code="VOCAB_CONFIG_INVALID",
        ) from cause

    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as cause:
        raise ConfigError.from_cause(
            f"vocab config is not valid YAML: '{path}'",
            cause,
            error_code="VOCAB_CONFIG_INVALID",
        ) from cause

    try:
        return VocabBuildConfig.model_validate(raw_data)
    except ValidationError as cause:
        raise ConfigError.from_cause(
            f"vocab config does not satisfy the schema: '{path}'",
            cause,
            error_code="VOCAB_CONFIG_INVALID",
        ) from cause
