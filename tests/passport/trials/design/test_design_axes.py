"""Mirror tests of the axes of the folded design."""

import math

import pytest

from iec.passport.trials.design.design_axes import COLUMN_PREFIX, DENSITY, RULE_COUNT, RULE_NUMBER
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin

LAYOUT = BinLayout(
    volume_bounds=doubling_bounds(20, 80),
    density_bounds=doubling_bounds(1, 16),
    row_bounds=doubling_bounds(1, 8),
    density_reference=locate_bin(4, doubling_bounds(1, 16)),
)


def test_the_axes_name_their_columns_and_reference_bins_as_the_passport() -> None:
    assert COLUMN_PREFIX == {"P": "p_bin_", "M": "m_bin_", "R": "r_bin_"}
    references = (DENSITY.reference(LAYOUT), RULE_COUNT.reference(LAYOUT))
    assert (*references, RULE_NUMBER.reference(LAYOUT)) == (3, 1, 1)


def test_the_axes_give_bins_bounds_and_abscissae_in_their_units() -> None:
    assert (DENSITY.bin_of(LAYOUT, 4, 1), DENSITY.bounds(3, LAYOUT)) == (3, (0.25, 0.5))
    assert (DENSITY.coordinate_sum(4, 1, 8), DENSITY.abscissa(0.25)) == (2.0, {"p": 0.25})
    assert (RULE_COUNT.bin_of(LAYOUT, 4, 3), RULE_COUNT.bounds(3, LAYOUT)) == (3, (3.0, 4.0))
    assert RULE_COUNT.abscissa(math.log(3.0)) == pytest.approx({"m": 3.0})
    assert (RULE_NUMBER.bin_of(LAYOUT, 6), RULE_NUMBER.bounds(9, LAYOUT)) == (4, (5.0, 8.0))
    assert RULE_NUMBER.coordinate_sum(2, 3) == pytest.approx(3 * math.log(2))
