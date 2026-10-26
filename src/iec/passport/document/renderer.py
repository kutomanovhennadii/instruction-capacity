"""The passport document: the order of the sections, the text, writing the documents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from iec.passport.document.figures import write_passport_figures
from iec.passport.document.labels import SECTION_TITLES
from iec.passport.document.sections import (
    bin_cube_lines,
    capacities_lines,
    collapse_lines,
    diagnostics_lines,
    figures_lines,
    header_lines,
    model_ids,
    model_params_lines,
)
from iec.passport.measurement_files import FIGURES_DIRNAME, PASSPORT_FILENAME

#: Sections of the passport written per model: the heading once, a subsection per model.
_PER_MODEL = ("capacities", "figures", "model_params", "diagnostics")


def render_passport(passport: Mapping[str, Any], figures: Mapping[str, Path]) -> str:
    """Return the text of ``passport.md``: the totals, the sections of every model, the
    collapse of the answer, the cube.

    A model section has its heading once and a subsection ``### `<model>``` for every model
    of the passport, in the order of the passport.
    """
    renders: dict[str, Callable[[str], list[str]]] = {
        "capacities": lambda model: capacities_lines(passport, model),
        "figures": lambda model: figures_lines(figures, model),
        "model_params": lambda model: model_params_lines(passport, model),
        "diagnostics": lambda model: diagnostics_lines(passport, model),
    }
    blocks: list[tuple[str, list[str]]] = [("header", header_lines(passport))]
    for key in _PER_MODEL:
        body: list[str] = []
        for model in model_ids(passport):
            part = renders[key](model)
            if part:
                body += ([""] if body else []) + [f"### `{model}`", "", *part]
        blocks.append((key, body))
    blocks.append(("collapse", collapse_lines(passport, figures)))
    blocks.append(("bin_cube", bin_cube_lines(passport)))
    lines = [f"# {SECTION_TITLES['document']} `{passport['model']['experiment_id']}`", ""]
    for key, body in blocks:
        if body:
            lines += [f"## {SECTION_TITLES[key]}", "", *body, ""]
    return "\n".join(lines).rstrip() + "\n"


def write_documents(
    passport: Mapping[str, Any], directory: Path
) -> tuple[Path, dict[str, Path]]:
    """Write the figures of every model, the collapse and ``passport.md`` into a directory.

    Returns:
        The path of ``passport.md`` and the written figures by file name.
    """
    figures = (
        write_passport_figures(
            dict(passport), directory / FIGURES_DIRNAME, str(passport["model"]["id"])
        )
        or {}
    )
    path = directory / PASSPORT_FILENAME
    path.write_text(render_passport(passport, figures), encoding="utf-8")
    return path, figures
