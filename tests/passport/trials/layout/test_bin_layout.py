"""Mirror tests of the bin layout and the doubling rule."""

import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.layout.bin_layout import BinLayout, doubling_bounds, locate_bin


def test_doubling_bounds_double_the_upper_bound() -> None:
    assert doubling_bounds(1, 10) == ((1, 1), (2, 2), (3, 4), (5, 8), (9, 10))
    assert doubling_bounds(20, 100) == ((1, 20), (21, 40), (41, 80), (81, 100))


def test_a_value_past_the_last_bin_is_pooled_into_it() -> None:
    bounds = doubling_bounds(1, 8)
    layout = BinLayout(
        volume_bounds=doubling_bounds(20, 80),
        density_bounds=bounds,
        row_bounds=bounds,
        density_reference=3,
    )

    assert (locate_bin(5, bounds), locate_bin(100, bounds)) == (4, 4)
    assert (layout.row_bin(3), layout.volume_bin(500), layout.R, layout.P) == (3, 3, 4, 4)
    with pytest.raises(ContractViolationError) as caught:
        layout.row_bin(0)
    assert caught.value.error_code == "BIN_LAYOUT_INVALID"
