"""The assignment of the forms of laws: the candidate forms of every block and the forms of the spread.

The assignment is a YAML file, ``config/passport_laws.yaml``, validated by a Pydantic model that
forbids unknown keys (specification §7.6). For every quantity it names the forms of every block,
with the options of a form by its name, and the form of the approximation of the variance of the
request shift by every axis ``V``, ``P`` and ``M``. The first form of a block is its established
form, the others are its alternatives, which replace it only beyond a threshold
(``iec.passport.laws.block_choice``); a block of one form - the slices of the grid - takes it.
``build_law_assignment`` builds the forms by the
registry once, at the setup of a command; a block asks its forms by the name of its quantity and its
own name. ``format_version`` is the version of the file format: version 2 is the only one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from iec.infrastructure.configuration import load_yaml_model
from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.laws.forms.form_registry import build_form
from iec.passport.laws.forms.law_form import LawForm

#: Error code of an assignment file that cannot be used.
LAW_ASSIGNMENT_CONFIG_INVALID = "LAW_ASSIGNMENT_CONFIG_INVALID"
#: Error code of a block or an axis that the assignment does not name.
LAW_BLOCK_UNKNOWN = "LAW_BLOCK_UNKNOWN"
#: Error code of a form that the consumer of a block does not draw: a section or the spread.
LAW_FORM_UNSUPPORTED = "LAW_FORM_UNSUPPORTED"


class BlockForms(BaseModel):
    """The law of one block: the names of its candidate forms and the options of a form by its name."""

    model_config = ConfigDict(extra="forbid")

    forms: list[str] = Field(min_length=1)
    options: dict[str, dict[str, float]] = Field(default_factory=dict)


class QuantityLaws(BaseModel):
    """The laws of one quantity: the forms of its blocks and of the spread by the axes."""

    model_config = ConfigDict(extra="forbid")

    blocks: dict[str, BlockForms]
    spread: dict[Literal["V", "P", "M"], str]


class LawAssignmentConfig(BaseModel):
    """The assignment file: the laws of every quantity of the passport."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[2] = 2
    quantities: dict[str, QuantityLaws]


def load_law_assignment_config(path: Path) -> LawAssignmentConfig:
    """Load and validate the assignment of the forms from a YAML file.

    Args:
        path: Path of the YAML file, for example ``config/passport_laws.yaml``.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: the file is missing or unreadable, the YAML is invalid or the schema is
            violated (``error_code="LAW_ASSIGNMENT_CONFIG_INVALID"``); the cause is kept.
    """
    return load_yaml_model(
        path,
        LawAssignmentConfig,
        error_code=LAW_ASSIGNMENT_CONFIG_INVALID,
        description="law assignment config",
    )


@dataclass(frozen=True, slots=True, eq=False)
class LawAssignment:
    """The forms of the laws of the passport, built once from the assignment file.

    ``blocks`` - the candidate forms of a block by ``(quantity, block)``, in the order of the file;
    ``spread`` - the form of the approximation of the variance by ``(quantity, axis)``.
    """

    blocks: Mapping[tuple[str, str], tuple[LawForm, ...]]
    spread: Mapping[tuple[str, str], LawForm]

    def forms_of(self, quantity: str, block: str) -> tuple[LawForm, ...]:
        """Return the candidate forms of a block.

        Raises:
            ContractViolationError: ``LAW_BLOCK_UNKNOWN`` - the assignment names no such block.
        """
        forms = self.blocks.get((quantity, block))
        if forms is None:
            raise ContractViolationError(
                f"the law assignment names no block '{block}' of {quantity}",
                error_code=LAW_BLOCK_UNKNOWN,
            )
        return forms

    def form_of(self, quantity: str, block: str) -> LawForm:
        """Return the form of a block of one form, or the established form of a chosen block.

        Raises:
            ContractViolationError: ``LAW_BLOCK_UNKNOWN`` - the assignment names no such block.
        """
        return self.forms_of(quantity, block)[0]

    def spread_form_of(self, quantity: str, axis: str) -> LawForm:
        """Return the form of the approximation of the variance by an axis.

        Raises:
            ContractViolationError: ``LAW_BLOCK_UNKNOWN`` - the assignment names no such axis.
        """
        form = self.spread.get((quantity, axis))
        if form is None:
            raise ContractViolationError(
                f"the law assignment names no spread form of the axis '{axis}' of {quantity}",
                error_code=LAW_BLOCK_UNKNOWN,
            )
        return form


def build_law_assignment(config: LawAssignmentConfig) -> LawAssignment:
    """Build the forms of every block and every axis of the spread by the registry.

    Args:
        config: The validated assignment.

    Returns:
        ``LawAssignment``.

    Raises:
        ConfigError: an unknown form, an option the form does not take or a value it rejects
            (``LAW_FORM_UNKNOWN``, ``LAW_FORM_OPTIONS_INVALID``, ``LAW_FORM_INVALID``).
    """
    blocks: dict[tuple[str, str], tuple[LawForm, ...]] = {
        (quantity, block): tuple(build_form(name, law.options.get(name)) for name in law.forms)
        for quantity, laws in config.quantities.items()
        for block, law in laws.blocks.items()
    }
    spread: dict[tuple[str, str], LawForm] = {
        (quantity, axis): build_form(name)
        for quantity, laws in config.quantities.items()
        for axis, name in laws.spread.items()
    }
    return LawAssignment(blocks, spread)
