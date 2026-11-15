"""The weighting of the points of a law as a function of the points kept in it.

A law goes through the bins of its block, and the outlier clipping (``sigma_clipping``) drops some
of them and fits the form again through the rest. The weight matrix of a fit belongs to its points,
so a section gives a law not one weighting but its rule: the weighting of any subset of the points
by their positions in the block. The rules are those of ``law_fit``: the full covariance of the
bins (generalised least squares), their own variances, or none.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iec.passport.laws.forms.law_fit import (
    LawWeighting,
    covariance_weighting,
    plain_weighting,
    variance_weighting,
)

#: The weighting of the points at the given positions of the block.
Weigh = Callable[[np.ndarray], LawWeighting]


def covariance_weights(block: np.ndarray) -> Weigh:
    """Return the rule of generalised least squares over the covariance of the points of a block."""
    matrix = np.asarray(block, dtype=float)
    return lambda kept: covariance_weighting(matrix[np.ix_(kept, kept)])


def variance_weights(variances: np.ndarray) -> Weigh:
    """Return the rule of the weights by the own variances of the points of a block."""
    own = np.asarray(variances, dtype=float)
    return lambda kept: variance_weighting(own[kept])


def plain_weights() -> Weigh:
    """Return the rule of plain least squares: every point weighs one."""
    return lambda kept: plain_weighting(len(kept))
