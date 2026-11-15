"""The threshold between the established form of a law and its alternatives."""

from __future__ import annotations

import pytest

from iec.passport.laws.form_threshold import (
    SWITCH_DELTA,
    Contender,
    overdispersion,
    pick,
    quasi_criterion,
)


def test_the_overdispersion_comes_from_the_most_flexible_form_and_is_not_below_one() -> None:
    line, hyperbola = Contender(0, 30.0, 2), Contender(1, 8.0, 4)
    assert overdispersion([line, hyperbola], 8) == pytest.approx(2.0)
    assert overdispersion([line, Contender(1, 1.0, 4)], 8) == pytest.approx(1.0)
    # A form without a degree of freedom gives no overdispersion.
    assert overdispersion([Contender(0, 3.0, 4)], 4) == pytest.approx(1.0)


def test_the_criterion_divides_the_chi2_by_the_overdispersion() -> None:
    assert quasi_criterion(Contender(0, 30.0, 2), 2.0) == pytest.approx(19.0)


def test_the_established_form_stays_within_the_threshold_and_yields_beyond_it() -> None:
    near = [Contender(0, 30.0, 2), Contender(1, 8.0, 4)]
    assert pick(near, 8, 0) == (0, pytest.approx(2.0))
    far = [Contender(0, 70.0, 2), Contender(1, 8.0, 4)]
    chosen, c_hat = pick(far, 8, 0)
    assert chosen == 1
    assert quasi_criterion(far[0], c_hat) - quasi_criterion(far[1], c_hat) > SWITCH_DELTA


def test_without_the_established_form_the_smallest_criterion_wins() -> None:
    assert pick([Contender(1, 12.0, 2), Contender(2, 2.0, 3)], 8, None)[0] == 2
