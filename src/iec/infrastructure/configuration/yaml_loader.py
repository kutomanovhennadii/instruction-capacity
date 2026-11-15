"""Loading of a YAML file of the bench: a configuration into a Pydantic model, or a plain mapping.

Every external configuration of the bench is YAML validated by a Pydantic model that forbids
unknown keys (specification §7.6). A data file of a free form, such as the version reference of a
vocabulary, is YAML whose top level is a mapping. This module is the one place that reads such
files: a missing file, invalid YAML, a violated schema and a top level that is not a mapping all
become a ``ConfigError`` with the error code of the file and the original cause kept.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from iec.infrastructure.exceptions import ConfigError

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(
    path: Path,
    model: type[ModelT],
    *,
    error_code: str,
    description: str,
) -> ModelT:
    """Load a YAML file and validate it with a Pydantic model.

    Args:
        path: Path of the YAML file.
        model: Pydantic model of the configuration.
        error_code: Error code of this configuration, for example ``VOCAB_CONFIG_INVALID``.
        description: Name of the configuration in error messages, for example ``vocab config``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: the file is missing or unreadable, the YAML is invalid or the schema is
            violated; the original cause is kept in ``__cause__``.
    """
    raw_data = _read_yaml(path, error_code=error_code, description=description)
    try:
        return model.model_validate(raw_data)
    except ValidationError as cause:
        raise ConfigError.from_cause(
            f"{description} does not satisfy the schema: '{path}'", cause, error_code=error_code
        ) from cause


def load_yaml_mapping(path: Path, *, error_code: str, description: str) -> dict[str, Any]:
    """Load a YAML file whose top level is a mapping, without a schema.

    Args:
        path: Path of the YAML file.
        error_code: Error code of this file, for example ``VOCAB_REFERENCE_INVALID``.
        description: Name of the file in error messages, for example ``vocab reference``.

    Returns:
        The mapping of the top level as YAML gives it.

    Raises:
        ConfigError: the file is missing or unreadable or the YAML is invalid, with the original
            cause kept in ``__cause__``; or the top level is not a mapping.
    """
    raw_data = _read_yaml(path, error_code=error_code, description=description)
    if not isinstance(raw_data, dict):
        raise ConfigError(f"{description} must be a mapping: '{path}'", error_code=error_code)
    return raw_data


def _read_yaml(path: Path, *, error_code: str, description: str) -> Any:
    """Read a YAML file and parse it; a failed read and invalid YAML become ``ConfigError``.

    Args:
        path: Path of the YAML file.
        error_code: Error code of the file.
        description: Name of the file in error messages.

    Returns:
        The parsed value: a mapping, a list, a scalar or ``None`` for an empty file.

    Raises:
        ConfigError: the file is missing or unreadable, or the YAML is invalid; the original
            cause is kept in ``__cause__``.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as cause:
        raise ConfigError.from_cause(
            f"{description} not found or unreadable: '{path}'", cause, error_code=error_code
        ) from cause
    try:
        return yaml.safe_load(raw_text)
    except yaml.YAMLError as cause:
        raise ConfigError.from_cause(
            f"{description} is not valid YAML: '{path}'", cause, error_code=error_code
        ) from cause
