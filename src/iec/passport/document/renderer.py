"""The passport document: ``passport.md`` from the passport dictionary.

The document opens with the part that a user reads: the capacities of the three quantities and the
standard stress test. Then comes the part that a specialist checks: the formula that combines the
three quantities and, for every quantity, its fit, its laws, its level and the figures. It computes
nothing: every number comes from the dictionary. Every quantity has its own renderer (``sections``);
the names of its figures come from the plan of the figures, the same plan that draws them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from iec.infrastructure.files import write_text_atomically
from iec.passport.document.charts.figure_plan import FigurePlan
from iec.passport.document.charts.figure_writer import figure_names, write_passport_figures
from iec.passport.document.sections.background_text import background_lines
from iec.passport.document.sections.capacities_text import capacities_lines
from iec.passport.document.sections.rule_word_text import rule_word_lines
from iec.passport.document.sections.stress_test_text import stress_test_lines
from iec.passport.document.sections.sync_text import sync_lines
from iec.passport.laws.forms.law_assignment import LawAssignment
from iec.passport.measurement_files import FIGURES_DIRNAME, PASSPORT_FILENAME

#: The renderers of the sections in the order of the document.
SECTION_RENDERERS: tuple[tuple[str, Callable[[Mapping[str, Any], Sequence[str]], list[str]]], ...] = (
    ("P_sync", sync_lines),
    ("H", rule_word_lines),
    ("F", background_lines),
)


def render_passport(passport: Mapping[str, Any], plan: FigurePlan) -> str:
    """Return the text of ``passport.md``, the part for a user first.

    Args:
        passport: The passport dictionary.
        plan: The plan of the figures: the names of the figures of every quantity.
    """
    lines = [
        f"# Passport of `{passport['model']['id']}`",
        "",
        f"Measurement `{passport['model']['experiment_id']}`, {passport['points']:,} answered points. "
        "The method is described in `docs/05-passport-model.md` of the repository.",
        "",
        *capacities_lines(passport),
    ]
    if "stress_test" in passport:
        lines += stress_test_lines(passport["stress_test"])
    lines += [
        "## The formula",
        "",
        "`P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`. The formula is the way to combine the three "
        "quantities: it has no law and no capacity of its own. `π = m/D` is the replacement density.",
        "",
    ]
    for name, render in SECTION_RENDERERS:
        section = passport[name]
        lines += render(section, figure_names(plan, name, section))
    return "\n".join(lines).rstrip() + "\n"


def write_documents(
    passport: Mapping[str, Any], directory: Path, assignment: LawAssignment, plan: FigurePlan
) -> tuple[Path, tuple[Path, ...]]:
    """Write the figures and ``passport.md`` into a directory.

    Args:
        passport: The passport dictionary.
        directory: The directory of the passport; the figures go into its ``figures/``.
        assignment: The forms of the laws: the curves of the figures.
        plan: The plan of the figures.

    Returns:
        The path of ``passport.md`` and the paths of the written figures.

    Side effects:
        Writes ``passport.md`` and the PNG files, each through a temporary file and a rename.
    """
    figures = write_passport_figures(passport, directory / FIGURES_DIRNAME, assignment, plan)
    path = directory / PASSPORT_FILENAME
    write_text_atomically(path, render_passport(passport, plan))
    return path, figures
