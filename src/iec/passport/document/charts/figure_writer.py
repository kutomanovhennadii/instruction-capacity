"""The figures of the passport by the plan: every entry of a quantity drawn by its kind.

The plan says which figures a quantity gets and in which order; a kind of figure draws an entry with
the law of its block by the form of the assignment of forms. The names that the document links come
from the same kinds, so the document and the folder of the figures cannot disagree. A quantity without
laws gets no figures.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from iec.passport.document.charts.chart_kinds import CHART_KINDS
from iec.passport.document.charts.figure_plan import FigurePlan
from iec.passport.laws.forms.law_assignment import LawAssignment

#: The status of a section that has laws and therefore figures.
MEASURED = "measured"


def figure_names(plan: FigurePlan, quantity: str, section: Mapping[str, Any]) -> list[str]:
    """Return the names of the figures of a quantity by the plan, without the extension.

    Args:
        plan: The plan of the figures.
        quantity: The quantity: ``P_sync``, ``H`` or ``F``.
        section: The section of the quantity in the passport dictionary.

    Returns:
        The names in the order of the plan; none for a quantity without laws.
    """
    if section.get("status") != MEASURED:
        return []
    return [name for entry in plan.of(quantity) for name in CHART_KINDS[entry.kind].names(entry, section)]


def write_passport_figures(
    passport: Mapping[str, Any], out_dir: Path, assignment: LawAssignment, plan: FigurePlan
) -> tuple[Path, ...]:
    """Draw the figures of the passport by the plan.

    Args:
        passport: The passport dictionary.
        out_dir: The folder of the figures; the subfolder of every quantity is created when needed.
        assignment: The forms of the laws: the curves of the figures.
        plan: The plan of the figures.

    Returns:
        The paths of the written files, in the order of the plan.

    Raises:
        ContractViolationError: ``FIGURE_VIEW_UNKNOWN`` - the plan names a block that no view draws;
            ``LAW_BLOCK_UNKNOWN`` - the assignment of forms names no such block.

    Side effects:
        Writes PNG files into the subfolders of ``out_dir``, each through a temporary file.
    """
    head = str(passport["model"]["id"])
    written: list[Path] = []
    for quantity, entries in plan.entries.items():
        section = passport[quantity]
        if section.get("status") != MEASURED:
            continue
        folder = out_dir / quantity
        folder.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            written += CHART_KINDS[entry.kind].draw(entry, quantity, section, assignment, head, folder)
    return tuple(written)
