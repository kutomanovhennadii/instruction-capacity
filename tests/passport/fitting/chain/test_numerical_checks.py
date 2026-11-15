"""The numerical checks: numbers that are not finite, negative variances, the information."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from iec.passport.fitting.chain.fit_notes import (
    INFORMATION_INDEFINITE,
    INFORMATION_UNSUPPORTED,
    NUMBERS_NOT_FINITE,
    VARIANCE_NEGATIVE,
)
from iec.passport.fitting.chain.numerical_checks import NUMERICAL_CHECKS, check_fit


@dataclass(frozen=True)
class _Evidence:
    """The estimates of two coefficients with a variance of the request shift."""

    theta: np.ndarray
    covariance: np.ndarray
    sigma2: float | None
    sigma2_row: np.ndarray
    information: np.ndarray


CLEAN = _Evidence(
    theta=np.array([0.5, -1.0]),
    covariance=np.array([[0.04, 0.01], [0.01, 0.09]]),
    sigma2=1.5,
    sigma2_row=np.array([0.001, 0.002, 0.03]),
    information=np.array([[30.0, 2.0, 0.5], [2.0, 15.0, 0.1], [0.5, 0.1, 8.0]]),
)


def _codes(evidence: _Evidence) -> list[str]:
    """Return the codes of the findings of the default checks."""
    return [note.code for note in check_fit(evidence)]


def test_clean_estimates_give_no_note() -> None:
    assert check_fit(CLEAN) == ()
    assert len(NUMERICAL_CHECKS) == 3


def test_a_number_that_is_not_finite_is_one_note_and_the_spectrum_is_not_read() -> None:
    information = CLEAN.information.copy()
    information[0, 1] = np.nan

    assert _codes(replace(CLEAN, information=information)) == [NUMBERS_NOT_FINITE]
    assert _codes(replace(CLEAN, sigma2=float("inf"))) == [NUMBERS_NOT_FINITE]


def test_a_negative_variance_is_noted() -> None:
    covariance = np.array([[-0.04, 0.01], [0.01, 0.09]])

    notes = check_fit(replace(CLEAN, covariance=covariance))

    assert [note.code for note in notes] == [VARIANCE_NEGATIVE]
    assert notes[0].message.startswith("1 of 3 variances")


def test_a_direction_without_support_is_noted() -> None:
    information = np.diag([30.0, 15.0, 1e-14])

    assert _codes(replace(CLEAN, information=information)) == [INFORMATION_UNSUPPORTED]


def test_a_negative_eigenvalue_is_noted() -> None:
    information = np.diag([30.0, 15.0, -2.0])

    assert _codes(replace(CLEAN, information=information)) == [INFORMATION_INDEFINITE]
