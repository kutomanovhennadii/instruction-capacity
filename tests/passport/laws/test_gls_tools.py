"""Generalised least squares through the bins: the points of a series and the sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.laws.gls_tools import delta_se, gls_least_squares, gls_sensitivity, named_points


def test_the_points_of_a_series_go_by_their_abscissae_and_skip_an_empty_bin() -> None:
    names = ["p_bin_3", "x", "p_bin_1", "p_bin_2"]
    covariance = np.diag([0.04, 1.0, 0.01, 0.0])
    theta = {"p_bin_3": 3.0, "x": 9.0, "p_bin_1": 1.0, "p_bin_2": 2.0}
    x, beta, block, index, kept = named_points(
        theta, covariance, names, "p_bin_", lambda number: float(number)
    )
    assert kept == ["p_bin_1", "p_bin_3"]
    assert x.tolist() == [1.0, 3.0]
    assert beta.tolist() == [1.0, 3.0]
    assert index == [2, 0]
    assert block.tolist() == [[0.01, 0.0], [0.0, 0.04]]


def test_a_line_is_recovered_and_its_sensitivity_is_the_projector() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    beta = 2.0 - 0.5 * x

    def curve(params: np.ndarray, at: np.ndarray) -> np.ndarray:
        result: np.ndarray = params[0] + params[1] * at
        return result

    def jacobian(params: np.ndarray, at: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(at), at])

    result, chi2 = gls_least_squares(
        curve, jacobian, np.zeros(2), np.full(2, -np.inf), np.full(2, np.inf), x, beta, np.eye(4)
    )
    assert result.x == pytest.approx([2.0, -0.5])
    assert chi2 == pytest.approx(0.0, abs=1e-12)
    design = jacobian(result.x, x)
    rows = gls_sensitivity(design, np.eye(4), [1, 2, 3, 4], 6)
    assert rows[:, 1:5] == pytest.approx(np.linalg.solve(design.T @ design, design.T))
    assert rows[:, [0, 5]].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert delta_se(np.array([3.0, 4.0]), np.eye(2)) == pytest.approx(5.0)
