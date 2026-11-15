"""The plan of the figures of the passport: which figures every quantity gets, in the order of the document.

The plan is a YAML file, ``config/passport_figures.yaml``, validated by a Pydantic model that forbids
unknown keys (specification §7.6). For every quantity it lists the entries of its figures in the order
of the document: the kind of a figure (``axis``, ``slices``, ``plateau``, ``spread``), the stem of the
names of its files and, for a figure of a law, the block of the assignment of forms whose law it draws.
The kinds draw the entries (``chart_kinds``), and the document links the same names
(``figure_writer.figure_names``). ``format_version`` is the version of the file format: version 1 is
the only one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from iec.infrastructure.configuration import load_yaml_model

#: Error code of a plan file that cannot be used.
FIGURE_PLAN_CONFIG_INVALID = "FIGURE_PLAN_CONFIG_INVALID"
#: The kinds of figures that draw the law of a block of the assignment of forms.
LAW_KINDS = ("axis", "slices", "plateau")


class FigureEntry(BaseModel):
    """One entry of the plan: a kind of figure, the stem of its files, the block whose law it draws."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["axis", "slices", "plateau", "spread"]
    stem: str
    block: str = ""

    @model_validator(mode="after")
    def _require_block(self) -> FigureEntry:
        """Require the block of a figure that draws the law of a block."""
        if self.kind in LAW_KINDS and not self.block:
            raise ValueError(f"a figure of the kind '{self.kind}' needs the block of its law")
        return self


class FigurePlanConfig(BaseModel):
    """The plan file: the entries of the figures of every quantity."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal[1] = 1
    quantities: dict[Literal["P_sync", "H", "F"], list[FigureEntry]]


@dataclass(frozen=True, slots=True, eq=False)
class FigurePlan:
    """The entries of the figures of every quantity in the order of the document."""

    entries: Mapping[str, tuple[FigureEntry, ...]]

    def of(self, quantity: str) -> tuple[FigureEntry, ...]:
        """Return the entries of a quantity; a quantity the plan does not name has none."""
        return self.entries.get(quantity, ())


def load_figure_plan(path: Path) -> FigurePlan:
    """Load and validate the plan of the figures from a YAML file.

    Args:
        path: Path of the YAML file, for example ``config/passport_figures.yaml``.

    Returns:
        The plan.

    Raises:
        ConfigError: the file is missing or unreadable, the YAML is invalid or the schema is
            violated (``error_code="FIGURE_PLAN_CONFIG_INVALID"``); the cause is kept.
    """
    config = load_yaml_model(
        path,
        FigurePlanConfig,
        error_code=FIGURE_PLAN_CONFIG_INVALID,
        description="figure plan config",
    )
    return FigurePlan({quantity: tuple(entries) for quantity, entries in config.quantities.items()})
