"""Mirror tests of the tables of the trials of a point."""

import math

import pytest

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.trial_tables import build_trial_tables


def test_the_tables_hold_the_doubling_bins_and_the_logarithms() -> None:
    tables = build_trial_tables(12)

    assert tables.top == 12
    assert tables.line_bins.tolist() == [0, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    assert tables.logarithms.tolist() == [0.0, *(math.log2(number) for number in range(1, 13))]


def test_a_number_past_the_top_is_an_error() -> None:
    tables = build_trial_tables(12)
    tables.require(12)

    with pytest.raises(ContractViolationError) as caught:
        tables.require(13)

    assert caught.value.error_code == "TRIAL_TABLES_INVALID"


def test_a_top_below_one_is_an_error() -> None:
    with pytest.raises(ContractViolationError) as caught:
        build_trial_tables(0)

    assert caught.value.error_code == "TRIAL_TABLES_INVALID"
