"""The weighting of a subset of the points of a law."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.laws.point_weights import covariance_weights, plain_weights, variance_weights


def test_the_covariance_rule_takes_the_block_of_the_kept_points() -> None:
    block = np.array([[4.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
    weighting = covariance_weights(block)(np.array([0, 2]))
    assert weighting.precision == pytest.approx(np.linalg.inv(block[np.ix_([0, 2], [0, 2])]))
    assert weighting.root @ weighting.root.T == pytest.approx(weighting.precision)


def test_the_variance_and_the_plain_rules_weigh_the_kept_points() -> None:
    weighting = variance_weights(np.array([4.0, 1.0, 0.25]))(np.array([1, 2]))
    assert np.diag(weighting.precision) == pytest.approx([1.0, 4.0])
    assert plain_weights()(np.array([0, 1, 2])).precision == pytest.approx(np.eye(3))
