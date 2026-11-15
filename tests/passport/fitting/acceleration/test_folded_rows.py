"""The folded rows: equal rows of a point fold into one, the points keep their order."""

from __future__ import annotations

import numpy as np

from iec.passport.fitting.acceleration.folded_rows import fold_rows


def test_equal_rows_of_a_point_fold_and_the_points_keep_their_order() -> None:
    x = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    y = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    run = np.array([1, 1, 1, 0, 0])

    rows = fold_rows(x, y, run)

    folded = {
        (int(point), tuple(row), trials, successes)
        for point, row, trials, successes in zip(
            rows.run, rows.x.tolist(), rows.trials.tolist(), rows.successes.tolist(), strict=True
        )
    }
    assert folded == {
        (0, (1.0, 0.0), 2.0, 2.0),
        (1, (0.0, 1.0), 1.0, 1.0),
        (1, (1.0, 0.0), 2.0, 1.0),
    }
    assert rows.run.tolist() == [0, 1, 1]
    assert rows.starts.tolist() == [0, 1]
    assert rows.n_runs == 2 and rows.n_trials == 5


def test_a_flat_feature_vector_is_one_column() -> None:
    rows = fold_rows(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0]), np.array([0, 0, 0]))

    assert rows.x.shape == (2, 1)
    assert rows.n_trials == 3
