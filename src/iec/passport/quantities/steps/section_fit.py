"""The first stage of a section: the design of a quantity without the rows of its fit, the fit, the support.

The laws of a quantity read of its design the bins of the axes, the cells of the grid and their abscissae,
and of its fit everything; the rows of the fit - the folded trials by coefficient - serve only the fit.
``light_design`` keeps the first and drops the second, so the first stage is small enough to be sent
between processes and saved (``passport.fit_store``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from iec.passport.fitting.variable_fit import VariableFit
from iec.passport.quantities.background_design import AnswerShareDesign
from iec.passport.trials.design.folded_design import VariableDesign


@dataclass(frozen=True, slots=True, eq=False)
class SectionFit:
    """The first stage of a quantity: its design without the rows of the fit, its fit and its support.

    ``support`` - the longest text of the measurement in words for P_sync and H, ``None`` for F.
    """

    design: VariableDesign | AnswerShareDesign
    fit: VariableFit
    support: float | None


def light_design(design: VariableDesign) -> VariableDesign:
    """Return the design without the rows of its fit and the keys of its requests."""
    return replace(
        design,
        x=design.x[:0],
        failures=design.failures[:0],
        trials=design.trials[:0],
        run=design.run[:0],
        point_keys=(),
    )
