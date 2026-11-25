"""The kinds of the figures of the passport behind one interface.

A kind names and draws the figures of one entry of the plan of a quantity: the bins of an axis with its
law before and after the shift correction, the cells of the grid by length slices with the curve of
every slice before and after, the levels of the slices with the law of the level, the variance of the
request shift by the groups of the three axes with its approximations. The curves and the labels come
from the views of the blocks (``law_views``) and from the drawings of the spread (``law_drawings``), the
frame from ``chart_frame``. The registry ``CHART_KINDS`` gives a kind by its name in the plan.

On an axis the whiskers are the own error of a bin, ±1.96·√q (the quasi-variance); a reference bin has
one too, its zero comes from the coding alone. On the cells of the grid the whiskers are ±1.96·SE of the
coefficient. An open point is the reference bin, a diamond is a bin outside the law.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iec.passport.document.charts.chart_frame import CURVE, finish  # noqa: E402
from iec.passport.document.charts.figure_plan import FigureEntry  # noqa: E402
from iec.passport.document.charts.law_views import (  # noqa: E402
    AXIS_VIEWS,
    PLATEAU_VIEWS,
    SLICE_VIEWS,
    AxisView,
    PlateauView,
    SliceView,
    known_view,
)
from iec.passport.document.law_drawings import SPREAD_SYMBOLS, spread_drawing  # noqa: E402
from iec.passport.document.number_text import number  # noqa: E402
from iec.passport.laws.forms.law_assignment import LawAssignment  # noqa: E402

#: The two states of a figure of an axis or of the grid: before and after the shift correction.
STATES = ((False, "before"), (True, "after"))
#: The titles of the quantities.
TITLES = {
    "P_sync": "P_sync: a word of the text stands on its place",
    "H": "H: a rule word on its place is replaced correctly",
    "F": "F: no misfire on a background word",
}
#: The marks of the bins of an axis by their status.
BIN_STYLES: Mapping[str, Mapping[str, Any]] = {
    "law": dict(fmt="o", color="black", ms=5, label="bin of the law"),
    "reference": dict(fmt="o", color="black", mfc="white", ms=7, label="reference bin"),
    "outlier": dict(
        fmt="D",
        color="#D55E00",
        mfc="white",
        ms=6,
        label="outside the law: an outlier beyond three sigma",
    ),
}
#: The axes of the figures of the spread: the letter, the name in the title, the label of the abscissa.
SPREAD_FIGURE_AXES = (
    ("V", "text length", "v = D·n: text length, words (centre of the group)"),
    ("P", "replacement density", "p = m/D: replacement density (centre of the group)"),
    ("M", "number of rules", "m: number of rules (centre of the group)"),
)


class ChartKind(Protocol):
    """A kind of figure: the names of the figures of a plan entry and their drawing."""

    def names(self, entry: FigureEntry, section: Mapping[str, Any]) -> list[str]:
        """Return the names of the figures of the entry, without the extension."""
        ...

    def draw(
        self,
        entry: FigureEntry,
        quantity: str,
        section: Mapping[str, Any],
        assignment: LawAssignment,
        head: str,
        folder: Path,
    ) -> list[Path]:
        """Draw the figures of the entry into the folder and return their paths."""
        ...


def _corrected_names(stem: str) -> list[str]:
    """Return the names of a figure before and after the shift correction."""
    return [f"{stem}-{state}-correction" for _, state in STATES]


def _state(shift: float, shifted: bool) -> str:
    """Return the state of a figure in its title: before the correction or after it."""
    return "after the correction: the shift " + number(shift) + " is subtracted" if shifted else "before the correction"


def _draw_axis(view: AxisView, shifted: bool, title: str, out: Path) -> None:
    """Draw the bins of an axis and its law in the coding of the fit or minus the shift of the axis."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    points = sorted((row for row in view.rows if "raw" in row), key=lambda row: row["center"])
    offset = view.shift if shifted else 0.0
    ax.plot(
        [row["center"] for row in points],
        [row["raw"] - offset for row in points],
        color="#999999",
        lw=0.9,
        zorder=2,
    )
    for status, style in BIN_STYLES.items():
        chosen = [row for row in points if row["status"] == status]
        if chosen:
            ax.errorbar(
                [row["center"] for row in chosen],
                [row["raw"] - offset for row in chosen],
                yerr=[row["own_half"] for row in chosen],
                capsize=3,
                zorder=5,
                **style,
            )
    ax.plot(view.grid, view.curve(view.grid) - offset, color=CURVE, lw=1.6, zorder=4, label=view.label)
    ylabel = "logit of a success minus the shift of the axis" if shifted else "logit of a success, coefficient of the bin"
    finish(
        ax,
        f"{title}\n{_state(view.shift, shifted)}; whiskers: own error of a bin, ±1.96·√q; lower means more errors",
        view.xlabel,
        ylabel,
        out,
        view.log_x,
    )


def _draw_slices(
    section: Mapping[str, Any], view: SliceView, shifted: bool, title: str, out: Path
) -> None:
    """Draw the cells of the grid by length slices and the law of every slice."""
    cells = [row for row in section["bins"]["grid"] if "raw" in row]
    numbers = sorted({row["v_bin"] for row in cells})
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.9, len(numbers)))
    offset = view.shift if shifted else 0.0
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    for color, v_bin in zip(colors, numbers, strict=True):
        members = sorted((row for row in cells if row["v_bin"] == v_bin), key=lambda row: row["x"])
        low, high = members[0]["words"]
        ax.errorbar(
            [row["x"] for row in members],
            [row["raw"] - offset for row in members],
            yerr=[row["raw_half"] for row in members],
            fmt="o-",
            color=color,
            ms=4,
            lw=1.1,
            capsize=2,
            label=f"text {low}–{high} words",
        )
        if v_bin in view.slices:
            grid = np.geomspace(1.0, max(row["x"] for row in members) * 1.2, 200)
            ax.plot(grid, view.slices[v_bin](grid) - offset, "--", color=color, lw=1.5)
    ax.plot([], [], "--", color="black", label=view.label)
    finish(
        ax,
        f"{title}\ncells by length slices, {_state(view.shift, shifted)}; whiskers ±1.96·SE",
        "s: word number by the reference (centre of mass of the cell)",
        "logit of a success of the cell; lower means more errors",
        out,
    )


def _draw_plateau(view: PlateauView, title: str, out: Path) -> None:
    """Draw the levels of the slices by the text length and the law of the level."""
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.errorbar(
        view.words,
        view.levels,
        yerr=view.halves,
        fmt="o-",
        color="black",
        ms=5,
        lw=0.9,
        capsize=3,
        label="level of a slice ± 1.96·SE",
    )
    ax.plot(view.grid, view.curve(view.grid), color=CURVE, lw=1.6, label=view.label)
    finish(
        ax,
        f"{title}\nlevels of the slices of the grid by the text length, before the correction",
        "v = D·n: text length, words",
        "level of a slice, logit of a success",
        out,
    )


@dataclass(frozen=True, slots=True)
class AxisChart:
    """The bins of an axis and its law, before and after the shift correction."""

    def names(self, entry: FigureEntry, section: Mapping[str, Any]) -> list[str]:
        """Return ``<stem>-before-correction`` and ``<stem>-after-correction``."""
        return _corrected_names(entry.stem)

    def draw(
        self,
        entry: FigureEntry,
        quantity: str,
        section: Mapping[str, Any],
        assignment: LawAssignment,
        head: str,
        folder: Path,
    ) -> list[Path]:
        """Draw the axis of the block of the entry with its law by the form of the block."""
        build = known_view(AXIS_VIEWS, quantity, entry.block)
        view = build(section, assignment.forms_of(quantity, entry.block))
        paths = [folder / f"{name}.png" for name in self.names(entry, section)]
        for (shifted, _), path in zip(STATES, paths, strict=True):
            _draw_axis(view, shifted, f"{head}: {TITLES[quantity]}: {view.name}", path)
        return paths


@dataclass(frozen=True, slots=True)
class SlicesChart:
    """The cells of the grid by length slices and the curve of every slice, before and after."""

    def names(self, entry: FigureEntry, section: Mapping[str, Any]) -> list[str]:
        """Return ``<stem>-before-correction`` and ``<stem>-after-correction``."""
        return _corrected_names(entry.stem)

    def draw(
        self,
        entry: FigureEntry,
        quantity: str,
        section: Mapping[str, Any],
        assignment: LawAssignment,
        head: str,
        folder: Path,
    ) -> list[Path]:
        """Draw the slices of the grid with the law of a slice by the form of the block."""
        build = known_view(SLICE_VIEWS, quantity, entry.block)
        view = build(section, assignment.forms_of(quantity, entry.block))
        paths = [folder / f"{name}.png" for name in self.names(entry, section)]
        for (shifted, _), path in zip(STATES, paths, strict=True):
            _draw_slices(section, view, shifted, f"{head}: {TITLES[quantity]}", path)
        return paths


@dataclass(frozen=True, slots=True)
class PlateauChart:
    """The levels of the slices by the text length and the law of the level, before the correction."""

    def names(self, entry: FigureEntry, section: Mapping[str, Any]) -> list[str]:
        """Return ``<stem>-before-correction``."""
        return [f"{entry.stem}-before-correction"]

    def draw(
        self,
        entry: FigureEntry,
        quantity: str,
        section: Mapping[str, Any],
        assignment: LawAssignment,
        head: str,
        folder: Path,
    ) -> list[Path]:
        """Draw the levels of the slices with the law of the level by the form of the block."""
        build = known_view(PLATEAU_VIEWS, quantity, entry.block)
        view = build(section, assignment.forms_of(quantity, entry.block))
        path = folder / f"{self.names(entry, section)[0]}.png"
        _draw_plateau(view, f"{head}: {TITLES[quantity]}", path)
        return [path]


@dataclass(frozen=True, slots=True)
class SpreadChart:
    """The variance of the request shift by the groups of the three axes and its approximations."""

    def names(self, entry: FigureEntry, section: Mapping[str, Any]) -> list[str]:
        """Return ``<stem>-V``, ``<stem>-P`` and ``<stem>-M``; a quantity with one variance has none."""
        if not section["fit"].get("spread"):
            return []
        return [f"{entry.stem}-{letter}" for letter, _, _ in SPREAD_FIGURE_AXES]

    def draw(
        self,
        entry: FigureEntry,
        quantity: str,
        section: Mapping[str, Any],
        assignment: LawAssignment,
        head: str,
        folder: Path,
    ) -> list[Path]:
        """Draw the variance by the groups of every axis with its approximation, if there is one."""
        spread = section["fit"].get("spread")
        if not spread:
            return []
        paths = []
        for letter, name, xlabel in SPREAD_FIGURE_AXES:
            rows = sorted(spread[letter], key=lambda row: float(row["center"]))
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            ax.errorbar(
                [float(row["center"]) for row in rows],
                [float(row["sigma2"]) for row in rows],
                yerr=[float(row["half"]) for row in rows],
                fmt="o-",
                color="black",
                ms=5,
                lw=0.9,
                capsize=3,
                label="group of bins",
            )
            law = section.get("spread_laws", {}).get(letter)
            if law is not None:
                drawing = spread_drawing(law)
                centers = np.array([float(row["center"]) for row in rows])
                grid = np.geomspace(centers.min(), centers.max(), 300)
                label = (
                    f"{drawing.formula(SPREAD_SYMBOLS[letter])}, χ² = {float(law['chi2']):.1f} "
                    f"over {law['n_points']} groups"
                )
                ax.plot(grid, drawing.curve(law, grid), color=CURVE, lw=1.6, label=label)
            path = folder / f"{entry.stem}-{letter}.png"
            finish(
                ax,
                f"{head}: variance of the request shift by {name}\nthe other two axes averaged over the "
                "requests; whiskers ±1.96·SE",
                xlabel,
                "variance of the request shift σ²",
                path,
            )
            paths.append(path)
        return paths


#: The kinds of figures by their names in the plan.
CHART_KINDS: Mapping[str, ChartKind] = {
    "axis": AxisChart(),
    "slices": SlicesChart(),
    "plateau": PlateauChart(),
    "spread": SpreadChart(),
}
