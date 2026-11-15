"""The bins of the passport axes as the fits of the three quantities use them.

Text length and word position form one grid of free cells: doubling the text costs much at the
start of the answer and almost nothing deep in it, so they are not a sum of two axes. Density bins
are numbered from the sparse end (1) to ``p = 1`` (the last). The grid of a layout is a few dozen
cells and is computed at every call: a cache of the process would be hidden state without a gain.
"""

from __future__ import annotations

import math

from iec.passport.trials.layout.bin_layout import BinLayout
from iec.passport.trials.layout.vpm_cube import density_level, density_p
from iec.prompt_generator.text_generator import WORDS_PER_LINE


def doub_bin(value: int) -> int:
    """Return the doubling bin of a value counted from one: ``{1}, {2}, {3,4}, {5..8}, ...``.

    The top of bin ``k`` is ``2^(k−1)``. The word-position bin of the grid is the doubling bin
    of the answer line, ``doub(⌈s/WORDS_PER_LINE⌉)``.
    """
    value = max(1, int(value))
    return 1 if value <= 1 else int(math.floor(math.log2(value - 1))) + 2


def vs_grid_tops(layout: BinLayout) -> tuple[int, ...]:
    """Return, per length bin, the last word-position bin of the answer.

    An answer word does not stand past the end of the text, so the position bin goes up to
    the bin of the last line of the longest text of the length bin:
    ``doub(⌈top of the bin / WORDS_PER_LINE⌉)``.
    """
    return tuple(doub_bin(-(-upper // WORDS_PER_LINE)) for _, upper in layout.volume_bounds)


def vs_grid_cells(layout: BinLayout) -> tuple[tuple[int, int], ...]:
    """Return the cells of the length × position grid in order: (length bin, position bin)."""
    return tuple(
        (volume, position)
        for volume, top in enumerate(vs_grid_tops(layout), start=1)
        for position in range(1, top + 1)
    )


def axis_density_bin(layout: BinLayout, D: int, m: int) -> int:  # noqa: N803
    """Return the density bin: from the sparse end (1) to ``p = 1`` (the last).

    The number is the run-cube density level (``0`` at ``p = 1``) shifted so that the
    sparsest level of the layout is ``1``: ``level + P``. A node sparser than the layout goes
    to the first bin - the same edge pooling as on the other axes.
    """
    return max(1, density_level(D, m) + layout.P)


def axis_density_reference(layout: BinLayout) -> int:
    """Return the number of the reference density bin.

    The layout keeps the reference bin ``k`` counted from ``p = 1``; its level is ``−(k−1)``,
    its number ``−(k−1) + P``.
    """
    return layout.P - layout.density_reference + 1


def axis_density_p(layout: BinLayout, number: int) -> tuple[float, float]:
    """Return the density of a bin ``[p_lo, p_hi]``; the last bin is ``[1, 1]``."""
    return density_p(number - layout.P)
