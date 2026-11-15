# acceleration — the structures that make the fits fast

The structures built once before the fits and handed to them ready (specification §2.11). Each is an object with its contract
and the key that tells when to rebuild it; the fits do not know how it was built.

## What is here

| Module | Structure | Key |
|---|---|---|
| `folded_rows.py` | `FoldedRows`: the equal rows of one point folded into one with their trials and successes - exact for the logit and the GLMM; `fold_rows()`; the protocol `FoldedDesign` of what a fit reads | the trials it was folded from |
| `sparse_design.py` | `SparseDesign`: the folded rows with the feature matrix in compressed sparse rows and the sums by point; `build_sparse_design()` | the folded rows it was built from |
| `quadrature_rule.py` | `QuadratureRule`: the Gauss-Hermite nodes, their weights and the weights of the adaptive form; `build_quadrature_rule()` | `size`, the number of nodes |

## Key signatures

```python
fold_rows(x: np.ndarray, y: np.ndarray, run: np.ndarray) -> FoldedRows
build_sparse_design(rows: FoldedDesign) -> SparseDesign
SparseDesign.run_sums(values) -> np.ndarray    # (n_runs, size)
SparseDesign.gram(values) -> np.ndarray        # (size, size)
build_quadrature_rule(size: int) -> QuadratureRule    # size < 1 -> ContractViolationError(QUADRATURE_RULE_INVALID)
```

## Who builds what

The rule of the marginal likelihood (15 nodes) and the rule of the threshold of a level (64 nodes) are built once, when
`glmm_fit` is loaded. The chain of a quantity folds its remaining rows once and builds their sparse design once; the plain
logit, the GLMM from every start and the fit with the spread by the load receive them ready.

## Boundaries

The folder holds the structures and the functions that build them; it fits nothing and keeps no state of the process.
