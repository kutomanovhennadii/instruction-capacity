"""The trial and the contract of a passport model: its name, coefficients, design row and fit.

The passport sees a model only through :class:`PassportModel` and builds a section for every
model it is given (:func:`iec.passport.passport_data.build_passport_data`). Models are
independent of each other and share nothing but this contract. The section, the second stage
and the public quantities read the coefficients by the names of the VS-PMR bins (``vs_V_S``,
``p_bin_k``, ``m_bin_k``, ``r_bin_k``): a passport model is a model of the VS-PMR formula, and
the way it fits the design is its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from iec.contracts import TrialRecord
from iec.passport.measured_points import MeasuredPoint

if TYPE_CHECKING:
    from iec.passport.bin_layout import BinLayout
    from iec.passport.design import Design
    from iec.passport.model_fitting import FitResult


@dataclass(frozen=True, slots=True)
class Trial:
    """Coordinates of one trial - everything the model formula may read.

    ``word_position`` is the position of the word in the text counting **from one**; the
    trial record keeps it from zero, and :func:`observed_trial` makes the shift.
    ``event_index`` is the number of the replacement in the point, from one (the validator's
    ``t``); VS-PMR does not read it.
    """

    D: int  # noqa: N815 - protocol axis name
    n: int
    m: int
    rule_index: int
    word_position: int
    event_index: int


def observed_trial(point: MeasuredPoint, record: TrialRecord) -> Trial:
    """Return the coordinates of an observed trial: the axes of its point and its record.

    The only place of the position shift: the record counts word positions from zero, the
    bins need a number from one. The event number ``t`` needs no shift.
    """
    return Trial(
        D=point.D,
        n=point.n,
        m=point.m,
        rule_index=record.rule_index,
        word_position=record.word_position + 1,
        event_index=record.t,
    )


class PassportModel(Protocol):
    """A passport model: its name, its coefficients, the design row of a trial, its fit."""

    @property
    def profile_id(self) -> str:
        """Name of the model - the key of its section in the passport."""
        ...

    def parameter_names(self, layout: BinLayout) -> tuple[str, ...]:
        """Return the names of the coefficients in the order of ``trial_vector``."""
        ...

    def trial_vector(
        self,
        trial: Trial,
        layout: BinLayout,
        reference_D: int,  # noqa: N803 - protocol axis name
        reference_n: int,
    ) -> tuple[float, ...]:
        """Return the design row of one trial."""
        ...

    def fit(
        self,
        design: Design,
        points: Sequence[MeasuredPoint],
        run_weights: Mapping[str, float] | None,
    ) -> FitResult:
        """Fit the design with point weights (``None`` - a plain fit).

        ``points`` are the points of the area of analysis: a model that estimates the variance
        of the point shift outside the fit takes its replicates from them.

        Raises:
            PassportError: ``PASSPORT_INPUT_MISSING`` - the design has no trials;
                ``FIT_NOT_CONVERGED`` - the fit did not converge.
        """
        ...
