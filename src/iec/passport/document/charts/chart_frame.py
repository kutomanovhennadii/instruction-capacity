"""The frame of a figure of the passport: the finish and the reliable write of the image.

Every figure has the zero line, a binary logarithmic abscissa unless the axis is a share, the grid, the
legend and the title. A figure is written whole or not at all: into a temporary file next to its place
and renamed onto it, as ``iec.infrastructure.files`` writes a text file, so a broken process leaves no
cut image.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iec.infrastructure.files.atomic_writing import TEMPORARY_SUFFIX  # noqa: E402

#: The colour of a law curve.
CURVE = "#d62728"


def save_figure(figure: Any, out: Path) -> None:
    """Write a figure as PNG into a temporary file next to its place and rename it onto its place.

    Args:
        figure: The figure of matplotlib.
        out: The path of the PNG file.

    Side effects:
        Creates or replaces ``out``; no temporary file stays after a successful write.
    """
    temporary = out.with_name(out.name + TEMPORARY_SUFFIX)
    figure.savefig(temporary, format="png")
    os.replace(temporary, out)


def finish(ax: Any, title: str, xlabel: str, ylabel: str, out: Path, log_x: bool = True) -> None:
    """Finish a figure and write it.

    Args:
        ax: The axes of the figure.
        title: The title.
        xlabel: The label of the abscissa.
        ylabel: The label of the ordinate.
        out: The path of the PNG file.
        log_x: A binary logarithmic abscissa.
    """
    ax.axhline(0.0, color="#cccccc", lw=0.8)
    if log_x:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.figure.tight_layout()
    save_figure(ax.figure, out)
    plt.close(ax.figure)
