"""The outlier clipping of a law by the algorithm of astropy."""

from __future__ import annotations

import numpy as np
import pytest

from iec.passport.laws.forms.law_fit import fit_law, plain_weighting
from iec.passport.laws.forms.linear_forms import LinearForm
from iec.passport.laws.point_weights import plain_weights, variance_weights
from iec.passport.laws.sigma_clipping import clip_residuals, clipped_fit

LINE = LinearForm()
X = np.arange(10, dtype=float)
#: A line with a small even scatter.
SCATTER = np.array([0.05, -0.04, 0.03, -0.06, 0.02, -0.03, 0.06, -0.02, 0.04, -0.05])


def test_a_residual_beyond_three_scales_of_the_others_is_clipped() -> None:
    residuals = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.02, -0.02, 5.0])
    clipped = clip_residuals(residuals, np.zeros(10, dtype=bool))
    assert np.flatnonzero(clipped).tolist() == [9]


def test_a_clipped_residual_stays_clipped() -> None:
    before = np.zeros(10, dtype=bool)
    before[0] = True
    clipped = clip_residuals(np.zeros(10), before)
    assert clipped[0] and clipped.sum() == 1


def test_a_law_through_its_bins_exactly_clips_nothing() -> None:
    # Its residuals are rounding noise: a scale of 1e-15 is not a scatter of the data.
    result = clipped_fit(LINE, X, 1.0 + 0.5 * X, plain_weights())
    assert result is not None and result.kept.all()


def test_the_outlier_leaves_the_law_and_the_rest_is_fitted_again() -> None:
    values = 1.0 + 0.5 * X + SCATTER
    values[4] += 10.0
    result = clipped_fit(LINE, X, values, plain_weights())
    assert result is not None
    assert np.flatnonzero(~result.kept).tolist() == [4]
    kept = np.flatnonzero(result.kept)
    alone = fit_law(LINE, X[kept], values[kept], plain_weighting(kept.size))
    assert result.fit.params == pytest.approx(alone.params)
    # The clipped bin has a zero column of the sensitivity: it carries nothing into the law.
    assert result.fit.sensitivity[:, 4] == pytest.approx(0.0)
    assert result.fit.sensitivity[:, kept] == pytest.approx(alone.sensitivity)


def test_the_length_law_of_f_of_gpt4o_mini_keeps_all_its_bins() -> None:
    # The eight bins of the length law of F of GPT-4o mini from the discussion of the forms (11.11).
    words = np.array([79.0, 154.0, 305.0, 595.0, 1144.0, 2183.0, 4306.0, 8477.0])
    values = np.array([13.40, 13.28, 9.29, 7.62, 7.15, 6.52, 6.21, 5.78])
    sigma = np.array([1.342, 0.967, 0.374, 0.200, 0.091, 0.036, 0.025, 0.053])
    x = np.log2(words / 20.0)
    result = clipped_fit(LINE, x, values, variance_weights(sigma**2))
    assert result is not None
    assert result.kept.all() and result.rounds == 1
    residuals = values - LINE.value(result.fit.params, x)
    # One scale for all the bins: the left bins lie far from the weighted line, but within 3 σ.
    assert float(np.std(residuals)) == pytest.approx(1.99, abs=0.05)
