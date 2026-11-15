"""The sparse design: the sums by point and the Gram matrix equal the dense ones."""

from __future__ import annotations

import numpy as np

from iec.passport.fitting.acceleration.folded_rows import FoldedRows
from iec.passport.fitting.acceleration.sparse_design import build_sparse_design

VALUES = np.array([0.5, -1.0, 2.0, 0.25, 1.5])


def _rows() -> FoldedRows:
    """Five rows of three points over four columns, most entries zero."""
    x = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    return FoldedRows(
        x=x,
        successes=np.array([1.0, 0.0, 2.0, 1.0, 3.0]),
        trials=np.array([2.0, 1.0, 2.0, 3.0, 4.0]),
        run=np.array([0, 0, 1, 2, 2], dtype=np.int64),
        starts=np.array([0, 2, 3], dtype=np.int64),
        n_runs=3,
    )


def test_the_sums_by_point_equal_the_dense_sums() -> None:
    rows = _rows()
    dense = np.zeros((3, 4))
    for row, point in enumerate(rows.run):
        dense[point] += VALUES[row] * rows.x[row]

    design = build_sparse_design(rows)

    assert design.size == 4 and design.n_runs == 3
    assert np.allclose(design.run_sums(VALUES), dense, rtol=0.0, atol=1e-15)


def test_the_gram_matrix_equals_the_dense_one() -> None:
    rows = _rows()

    gram = build_sparse_design(rows).gram(VALUES)

    assert np.allclose(gram, rows.x.T @ (rows.x * VALUES[:, None]), rtol=0.0, atol=1e-15)


def test_the_design_keeps_the_counts_of_the_rows_and_one_entry_per_non_zero() -> None:
    rows = _rows()

    design = build_sparse_design(rows)

    assert np.array_equal(design.successes, rows.successes)
    assert np.array_equal(design.trials, rows.trials)
    assert np.array_equal(design.starts, rows.starts)
    assert len(design.entry_rows) == int(np.count_nonzero(rows.x))
