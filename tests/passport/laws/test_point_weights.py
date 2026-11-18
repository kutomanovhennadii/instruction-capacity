"""The weighting of a subset of the points of a law."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.laws.point_weights import dersimonian_laird, plain_weights, variance_weights


def test_the_variance_and_the_plain_rules_weigh_the_kept_points() -> None:
    weighting = variance_weights(np.array([4.0, 1.0, 0.25]))(np.array([1, 2]))
    assert np.diag(weighting.precision) == pytest.approx([1.0, 4.0])
    assert plain_weights()(np.array([0, 1, 2])).precision == pytest.approx(np.eye(3))


def test_dersimonian_laird_of_a_constant_model_is_the_classic_estimate() -> None:
    # Two points 1 and 3 of unit variance around one mean: χ² = 2, k − 1 = 1, Σw − Σw²/Σw = 1.
    assert dersimonian_laird(np.ones((2, 1)), 2.0, np.ones(2)) == pytest.approx(1.0)


def test_a_law_through_its_points_has_no_scatter() -> None:
    design = np.column_stack([np.ones(4), np.arange(4.0)])
    assert dersimonian_laird(design, 0.0, np.full(4, 0.01)) == 0.0


def test_the_scatter_of_a_line_is_its_excess_chi2_over_the_trace() -> None:
    # The line through 0, 1, 0, 1 leaves the residuals −0.2, 0.6, −0.6, 0.2: χ² = 0.8/0.01 = 80; with
    # equal weights w = 100 the trace is w·(k − p) = 200, so τ² = (80 − 2)/200 = 0.39 - the residual
    # variance of the line 0.8/2 less the own variance 0.01.
    design = np.column_stack([np.ones(4), np.arange(4.0)])
    assert dersimonian_laird(design, 80.0, np.full(4, 0.01)) == pytest.approx(0.39)
