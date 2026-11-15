"""Mirror tests of the designs of the three quantities."""

from iec.passport.trials.design.quantity_designs import QUANTITY_DESIGNS, VARIABLE_AXES
from iec.passport.trials.place_trials import PointTrials

POINT = PointTrials(
    key="a",
    D=40,
    n=4,
    m=10,
    sync=((1, 10, 1, 21.8),),
    rule=((3, 2, 2, 1, 9.6),),
    background=((3, 15, 0, 72.0),),
    totals={},
)


def test_every_quantity_takes_its_own_trials_of_a_point() -> None:
    assert QUANTITY_DESIGNS["P_sync"].cells(POINT) == [(1, (), 10, 1, 21.8)]
    assert QUANTITY_DESIGNS["H"].cells(POINT) == [(3, (2,), 2, 1, 9.6)]
    assert QUANTITY_DESIGNS["F"].cells(POINT) == [(3, (), 15, 0, 72.0)]


def test_the_axes_of_the_quantities_are_those_of_the_passport() -> None:
    assert VARIABLE_AXES == {"P_sync": ("P", "M"), "H": ("P", "M", "R"), "F": ("P", "M")}
