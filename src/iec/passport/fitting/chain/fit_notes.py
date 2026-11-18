"""The notes of the fit of a quantity: the shunts of its chain and the findings of its checks.

A shunt is the rule of a non-standard situation of a fit (specification §12.8): a path of the
chain that gives no fit leaves the quantity with the fit reached before it. A numerical check
reports a number that should not be trusted and changes nothing. Every such place has a stable
code; a note carries the code, the text for a human and the severity of its log record. The table
of the places is in the README of the folder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

#: No rows or columns remain after the exclusion rule: the quantity has no fit.
FIT_NO_DATA = "FIT_NO_DATA"
#: The plain logit did not converge or gave non-finite estimates: the quantity has no fit.
LOGIT_FAILED = "LOGIT_FAILED"
#: The GLMM from one start of the variance gave no fit: the next start follows.
GLMM_START_FAILED = "GLMM_START_FAILED"
#: The GLMM gave no fit from any start: the quantity keeps the plain logit.
GLMM_FAILED = "GLMM_FAILED"
#: The variance of the request shift stays on its floor: the spread by the load is not fitted.
SPREAD_ON_FLOOR = "SPREAD_ON_FLOOR"
#: Every axis of the spread has one group: the spread cannot depend on the load.
SPREAD_ONE_GROUP = "SPREAD_ONE_GROUP"
#: The fit with the spread by the load gave no fit: the quantity keeps one variance.
SPREAD_FAILED = "SPREAD_FAILED"
#: The fit with the spread stopped on its floor: the group of the lowest spread joined its neighbour.
SPREAD_MERGED = "SPREAD_MERGED"
#: The variance by the load turned bins against the median request of the data: the quantity keeps
#: one variance.
SPREAD_AGAINST_DATA = "SPREAD_AGAINST_DATA"
#: A number of the estimates, of their covariance or of the information is not finite.
NUMBERS_NOT_FINITE = "NUMBERS_NOT_FINITE"
#: A variance of the estimates is negative: the passport takes its standard error as zero.
VARIANCE_NEGATIVE = "VARIANCE_NEGATIVE"
#: Directions of the information below the threshold of the pseudo-inverse get zero variance.
INFORMATION_UNSUPPORTED = "INFORMATION_UNSUPPORTED"
#: The information has negative eigenvalues: the estimates are not at a maximum.
INFORMATION_INDEFINITE = "INFORMATION_INDEFINITE"
#: Every code of the catalog.
NOTE_CODES: frozenset[str] = frozenset(
    {
        FIT_NO_DATA,
        LOGIT_FAILED,
        GLMM_START_FAILED,
        GLMM_FAILED,
        SPREAD_ON_FLOOR,
        SPREAD_ONE_GROUP,
        SPREAD_FAILED,
        SPREAD_MERGED,
        SPREAD_AGAINST_DATA,
        NUMBERS_NOT_FINITE,
        VARIANCE_NEGATIVE,
        INFORMATION_UNSUPPORTED,
        INFORMATION_INDEFINITE,
    }
)


@dataclass(frozen=True, slots=True)
class FitNote:
    """One note of the fit of a quantity.

    ``code`` - the place in the catalog of shunts; ``message`` - the text for a human; ``level`` -
    the severity of its log record: ``logging.INFO`` for a turn of the chain that its rule
    foresees, ``logging.WARNING`` for a fallback or a finding of a check.
    """

    code: str
    message: str
    level: int = logging.WARNING
