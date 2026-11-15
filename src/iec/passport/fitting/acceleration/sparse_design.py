"""The sparse design of the GLMM: the folded rows with a sparse feature matrix and sums by point.

At every Newton step the GLMM sums values of the rows of every point times their features. A
row has a grid cell and at most one bin of every axis, so it holds a few non-zeros out of dozens
of columns, and the sums go through the non-zeros only. The sparse design is a structure for
speed (specification §2.11): it is built once from the folded rows by ``build_sparse_design``,
and every fit on these rows receives it ready - the fits from several starts of the variance and
the fit with the spread by the load. Its key is the folded rows it was built from: other rows
need a new sparse design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from iec.passport.fitting.acceleration.folded_rows import FoldedDesign


@dataclass(frozen=True, slots=True, eq=False)
class SparseDesign:
    """The folded rows of the GLMM with the feature matrix in compressed sparse rows.

    ``successes``, ``trials``, ``run``, ``starts`` and ``n_runs`` - as in the folded rows;
    ``size`` - the number of columns; ``x`` - the feature matrix; ``entry_rows``,
    ``entry_values`` and ``entry_slots`` - the row, the value and the slot
    ``point · size + column`` of every non-zero of ``x``.
    """

    successes: np.ndarray
    trials: np.ndarray
    run: np.ndarray
    starts: np.ndarray
    n_runs: int
    size: int
    x: sparse.csr_matrix
    entry_rows: np.ndarray
    entry_values: np.ndarray
    entry_slots: np.ndarray

    def run_sums(self, values: np.ndarray) -> np.ndarray:
        """Return ``sum over the rows i of point j of values_i * x_i`` - ``(n_runs, size)``."""
        flat = np.bincount(
            self.entry_slots,
            weights=self.entry_values * values[self.entry_rows],
            minlength=self.n_runs * self.size,
        )
        return flat.reshape(self.n_runs, self.size)

    def gram(self, values: np.ndarray) -> np.ndarray:
        """Return ``sum over the rows i of values_i * x_i x_i'`` - ``(size, size)``."""
        return np.asarray((self.x.T @ self.x.multiply(values[:, None])).todense())


def build_sparse_design(rows: FoldedDesign) -> SparseDesign:
    """Build the sparse design of folded rows.

    Args:
        rows: The folded rows of a fit.

    Returns:
        ``SparseDesign`` with the same rows, the feature matrix in compressed sparse rows and
        the coordinates of its non-zeros.
    """
    run = np.asarray(rows.run)
    size = int(rows.x.shape[1])
    x = sparse.csr_matrix(rows.x)
    coo = x.tocoo()
    return SparseDesign(
        successes=np.asarray(rows.successes, dtype=float),
        trials=np.asarray(rows.trials, dtype=float),
        run=run,
        starts=np.asarray(rows.starts),
        n_runs=int(rows.n_runs),
        size=size,
        x=x,
        entry_rows=coo.row,
        entry_values=coo.data,
        entry_slots=run[coo.row] * size + coo.col,
    )
