"""Numerical checks of a fit: numbers that are not finite, negative variances, the information.

The checks look at the estimates of the reached path after the chain: the coefficients, the
variance of the request shift, their covariance and the information whose pseudo-inverse is the
bread of the sandwich. Three things make these numbers untrustworthy. A number that is not finite
comes from an overflow or a division by zero. A negative variance gets a zero standard error in
the passport. A direction of the information with an eigenvalue not above ``INFORMATION_RCOND``
of the largest one in absolute value is cut by the pseudo-inverse, and the covariance gives it
zero variance; a negative eigenvalue means that the estimates are not at a maximum.

The checks report and change nothing: every finding is a note of the fit and a warning in the
log. Every check is a strategy of one interface; the default set is ``NUMERICAL_CHECKS``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from iec.passport.fitting.chain.fit_notes import (
    INFORMATION_INDEFINITE,
    INFORMATION_UNSUPPORTED,
    NUMBERS_NOT_FINITE,
    VARIANCE_NEGATIVE,
    FitNote,
)
from iec.passport.fitting.logit_fit import INFORMATION_RCOND


class FitEvidence(Protocol):
    """What the checks read from the estimates of a fit."""

    @property
    def theta(self) -> np.ndarray:
        """Return the estimates of the coefficients."""
        ...

    @property
    def covariance(self) -> np.ndarray:
        """Return the covariance of the coefficients."""
        ...

    @property
    def sigma2(self) -> float | None:
        """Return the variance of the request shift, ``None`` without the shift."""
        ...

    @property
    def sigma2_row(self) -> np.ndarray:
        """Return the covariance row of the variance and its own variance at the end."""
        ...

    @property
    def information(self) -> np.ndarray:
        """Return the information whose pseudo-inverse is the bread of the sandwich."""
        ...


class NumericalCheck(Protocol):
    """One numerical check of the estimates of a fit."""

    def inspect(self, evidence: FitEvidence) -> tuple[FitNote, ...]:
        """Return the findings of the check; an empty tuple - nothing found."""
        ...


class FiniteNumbersCheck:
    """Every number of the estimates, of their covariance and of the information is finite."""

    def inspect(self, evidence: FitEvidence) -> tuple[FitNote, ...]:
        """Return a note with the count of the numbers that are not finite."""
        parts = [evidence.theta, evidence.covariance, evidence.sigma2_row, evidence.information]
        if evidence.sigma2 is not None:
            parts.append(np.array([evidence.sigma2]))
        arrays = [np.asarray(part, dtype=float) for part in parts]
        bad = sum(int(np.count_nonzero(~np.isfinite(array))) for array in arrays)
        if not bad:
            return ()
        total = sum(array.size for array in arrays)
        message = (
            f"{bad} of {total} numbers of the estimates, their covariance and the information "
            "are not finite"
        )
        return (FitNote(NUMBERS_NOT_FINITE, message),)


class NegativeVarianceCheck:
    """No variance of the coefficients and of the variance of the shift is negative."""

    def inspect(self, evidence: FitEvidence) -> tuple[FitNote, ...]:
        """Return a note with the count of the negative variances."""
        covariance = np.asarray(evidence.covariance, dtype=float)
        variances = np.diag(covariance) if covariance.size else np.zeros(0)
        row = np.asarray(evidence.sigma2_row, dtype=float)
        if row.size:
            variances = np.append(variances, row[-1])
        negative = int(np.count_nonzero(variances < 0.0))
        if not negative:
            return ()
        message = (
            f"{negative} of {variances.size} variances of the estimates are negative: "
            "their standard errors are taken as zero"
        )
        return (FitNote(VARIANCE_NEGATIVE, message),)


class InformationCheck:
    """The data support every direction of the information, and no eigenvalue is negative."""

    def inspect(self, evidence: FitEvidence) -> tuple[FitNote, ...]:
        """Return a note on the cut directions and a note on the negative eigenvalues.

        Information that is not finite is not inspected: the finiteness check reports it.
        """
        information = np.asarray(evidence.information, dtype=float)
        if information.size == 0 or not np.all(np.isfinite(information)):
            return ()
        values = np.linalg.eigvalsh(information)
        cutoff = INFORMATION_RCOND * float(np.max(np.abs(values)))
        notes: list[FitNote] = []
        cut = int(np.count_nonzero(np.abs(values) <= cutoff))
        if cut:
            message = (
                f"{cut} of {values.size} directions of the information are not above "
                f"{INFORMATION_RCOND:g} of its largest eigenvalue: the covariance gives them zero "
                "variance"
            )
            notes.append(FitNote(INFORMATION_UNSUPPORTED, message))
        negative = int(np.count_nonzero(values < -cutoff))
        if negative:
            message = (
                f"{negative} of {values.size} eigenvalues of the information are negative: "
                "the estimates are not at a maximum"
            )
            notes.append(FitNote(INFORMATION_INDEFINITE, message))
        return tuple(notes)


#: The checks every fit of a quantity goes through, in order.
NUMERICAL_CHECKS: tuple[NumericalCheck, ...] = (
    FiniteNumbersCheck(),
    NegativeVarianceCheck(),
    InformationCheck(),
)


def check_fit(
    evidence: FitEvidence, checks: Sequence[NumericalCheck] = NUMERICAL_CHECKS
) -> tuple[FitNote, ...]:
    """Return the findings of the checks in their order.

    Args:
        evidence: The estimates of the reached path.
        checks: The checks; by default ``NUMERICAL_CHECKS``.
    """
    return tuple(note for check in checks for note in check.inspect(evidence))
