"""Tables of the trials of a point: line number -> doubling bin, word number -> log2, built once.

The trials of a point need, for every place, the doubling bin of its line and the binary logarithm
of its word number (``place_trials``). A table answers both for a whole array of places at once.
The tables are a structure for speed (specification §2.11): they are built once at the setup of a
measurement for its longest text and handed to the counting of every point ready, so the counting
does not build them and no table lives in the process. Their key is ``top``, the largest number
they cover: a measurement with a longer text needs new tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from iec.infrastructure.exceptions import ContractViolationError
from iec.passport.trials.layout.axis_bins import doub_bin

#: Error code of tables that do not cover a number of a point, or of a top below one.
TRIAL_TABLES_INVALID = "TRIAL_TABLES_INVALID"


@dataclass(frozen=True, slots=True, eq=False)
class TrialTables:
    """The tables of the trials of the points of one measurement.

    ``line_bins[k]`` is the doubling bin of line ``k`` and ``logarithms[k]`` is ``log₂ k`` for
    ``k`` from 1 to ``top``; index 0 holds 0 in both. A line number never exceeds the word
    number, so one bound covers both tables.
    """

    line_bins: np.ndarray
    logarithms: np.ndarray

    @property
    def top(self) -> int:
        """Return the largest word number and line number the tables cover."""
        return len(self.logarithms) - 1

    def require(self, number: int) -> None:
        """Check that the tables cover a number.

        Args:
            number: The largest word number of a point.

        Raises:
            ContractViolationError: ``TRIAL_TABLES_INVALID`` - the number is past ``top``: the
                tables were built for a shorter text than the point has.
        """
        if number > self.top:
            raise ContractViolationError(
                f"the trial tables cover numbers up to {self.top}, a point needs {number}",
                error_code=TRIAL_TABLES_INVALID,
            )


def build_trial_tables(top: int) -> TrialTables:
    """Build the tables for word numbers and line numbers up to ``top``.

    Args:
        top: The length of the longest text of the measurement in words.

    Returns:
        ``TrialTables`` covering the numbers from 1 to ``top``.

    Raises:
        ContractViolationError: ``TRIAL_TABLES_INVALID`` - ``top`` is below one.
    """
    if top < 1:
        raise ContractViolationError(
            f"the trial tables need a top of at least 1, got {top}",
            error_code=TRIAL_TABLES_INVALID,
        )
    return TrialTables(
        line_bins=np.array([0, *(doub_bin(line) for line in range(1, top + 1))], dtype=np.int64),
        logarithms=np.array([0.0, *(math.log2(number) for number in range(1, top + 1))]),
    )
