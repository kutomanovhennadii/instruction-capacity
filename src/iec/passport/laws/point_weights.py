"""The weighting of the points of a law as a function of the points kept in it.

A law goes through the bins of its block, and the outlier clipping (``sigma_clipping``) drops some
of them and fits the form again through the rest. The weight matrix of a fit belongs to its points,
so a section gives a law not one weighting but its rule: the weighting of any subset of the points
by their positions in the block. The rules are the own variances of the bins or none; the scatter
``τ²`` of the points around their law beyond their own variances (``dersimonian_laird``) is added
to the own variances for the weights ``1/(σᵢ² + τ²)`` of the laws.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iec.passport.laws.forms.law_fit import LawWeighting, plain_weighting, variance_weighting

#: The weighting of the points at the given positions of the block.
Weigh = Callable[[np.ndarray], LawWeighting]


def variance_weights(variances: np.ndarray) -> Weigh:
    """Return the rule of the weights by the own variances of the points of a block."""
    own = np.asarray(variances, dtype=float)
    return lambda kept: variance_weighting(own[kept])


def plain_weights() -> Weigh:
    """Return the rule of plain least squares: every point weighs one."""
    return lambda kept: plain_weighting(len(kept))


def dersimonian_laird(jacobian: np.ndarray, chi2: float, variances: np.ndarray) -> float:
    """Return ``τ²``: the scatter of the points around their law beyond their own variances.

    The estimate of DerSimonian and Laird for a regression (Viechtbauer and Cheung 2010, formulas 9
    and 10), with the derivatives of the law by its free parameters in place of the design of a
    linear model: ``τ² = max(0, (χ² − (k − p))/tr P)``, ``P = W − WJ(JᵀWJ)⁺JᵀW``,
    ``W = diag(1/σᵢ²)``, ``χ²`` the weighted sum of the squared residuals of the law with the weights
    ``W``, ``k`` the number of the points and ``p`` the rank of ``J``. For a linear law ``χ² = yᵀPy``,
    and the estimate is that of the regression.

    Args:
        jacobian: The derivatives ``J`` of the law at the points by its free parameters.
        chi2: The weighted sum of the squared residuals of the law with the weights ``1/σᵢ²``.
        variances: The own variances ``σᵢ²`` of the points, positive.
    """
    weight = 1.0 / np.asarray(variances, dtype=float)
    matrix = np.asarray(jacobian, dtype=float)
    weighted = matrix * weight[:, None]
    hat = np.linalg.pinv(matrix.T @ weighted) @ (weighted.T @ weighted)
    trace = float(weight.sum() - np.trace(hat))
    if trace <= 0.0:
        return 0.0
    rank = int(np.linalg.matrix_rank(matrix))
    return max(0.0, (float(chi2) - (weight.size - rank)) / trace)
