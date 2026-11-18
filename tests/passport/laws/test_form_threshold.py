"""The choice among the forms of a law: the smallest criterion, the tie, the held parameter, ĉ."""

from __future__ import annotations

import pytest

from iec.passport.laws.form_threshold import (
    Contender,
    overdispersion,
    pick,
    quasi_criterion,
)


def test_the_overdispersion_is_the_smallest_ratio_and_not_below_one() -> None:
    # The line has 30/6 = 5 per degree of freedom, the hyperbola 8/4 = 2: the smaller is taken.
    line, hyperbola = Contender(0, 30.0, 2), Contender(1, 8.0, 4)
    assert overdispersion([line, hyperbola], 8) == pytest.approx(2.0)
    # 2/6 and 8/4: the smallest is a third, and the overdispersion is not below one.
    assert overdispersion([Contender(0, 2.0, 2), Contender(1, 8.0, 4)], 8) == pytest.approx(1.0)
    # A form without a degree of freedom gives no overdispersion.
    assert overdispersion([Contender(0, 3.0, 4)], 4) == pytest.approx(1.0)


def test_the_criterion_divides_the_chi2_by_the_overdispersion() -> None:
    assert quasi_criterion(Contender(0, 30.0, 2), 2.0) == pytest.approx(19.0)


def test_the_smallest_criterion_wins_and_the_established_form_takes_a_tie() -> None:
    # ĉ = 2: the criteria are 19 and 12, the alternative wins without a margin.
    assert pick([Contender(0, 30.0, 2), Contender(1, 8.0, 4)], 8) == (1, pytest.approx(2.0))
    assert pick([Contender(0, 8.0, 4), Contender(1, 8.0, 4)], 8)[0] == 0


def test_an_alternative_with_a_held_parameter_does_not_replace_the_established_form() -> None:
    # A parameter on a bound of its range keeps the established form.
    assert pick([Contender(0, 30.0, 2), Contender(1, 8.0, 4, held=True)], 8)[0] == 0
    # Without the established form among the contenders the held one competes.
    assert pick([Contender(1, 8.0, 4, held=True), Contender(2, 30.0, 2)], 8)[0] == 1


def test_without_the_established_form_the_smallest_criterion_wins() -> None:
    assert pick([Contender(1, 12.0, 2), Contender(2, 2.0, 3)], 8)[0] == 2
