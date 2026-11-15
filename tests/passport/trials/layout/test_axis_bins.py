"""Mirror tests of the bins of the axes as the fits use them."""

from iec.passport.trials.layout.axis_bins import (
    axis_density_bin,
    axis_density_p,
    axis_density_reference,
    doub_bin,
    vs_grid_cells,
    vs_grid_tops,
)
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 80),
    density_bounds=doubling_bounds(1, 16),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 16)),
)


def test_the_doubling_bin_counts_from_one() -> None:
    assert [doub_bin(value) for value in (0, 1, 2, 3, 4, 5, 8, 9)] == [1, 1, 2, 3, 3, 4, 4, 5]


def test_the_grid_goes_up_to_the_last_line_of_every_length_bin() -> None:
    # Length bins end at 20, 40 and 80 words: the last lines 2, 4 and 8 are in bins 2, 3 and 4.
    assert vs_grid_tops(LAYOUT) == (2, 3, 4)
    assert vs_grid_cells(LAYOUT)[:3] == ((1, 1), (1, 2), (2, 1))
    assert len(vs_grid_cells(LAYOUT)) == 9


def test_density_bins_count_from_the_sparse_end() -> None:
    # p = 1 is the last bin (P = 5), p = 1/4 is bin 3 - the reference bin of the start step.
    assert (axis_density_bin(LAYOUT, 8, 8), axis_density_bin(LAYOUT, 4, 1)) == (5, 3)
    assert (axis_density_bin(LAYOUT, 100, 1), axis_density_reference(LAYOUT)) == (1, 3)
    assert (axis_density_p(LAYOUT, 5), axis_density_p(LAYOUT, 3)) == ((1.0, 1.0), (0.25, 0.5))
