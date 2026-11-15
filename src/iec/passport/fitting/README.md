# fitting — the fits of the passport bins

The fits that turn the folded matrix of a quantity into the coefficients of its bins: the plain logit by Newton steps, the
GLMM with the shift of a request inside the likelihood, the spread of the shift by the load and the chain that tries them in
order for every quantity. The chain and its notes live in `chain/`, the structures that make the fits fast in `acceleration/`.

## What is here

| Module or folder | Role |
|---|---|
| `logit_fit.py` | the plain logit on folded rows by Newton steps, the cluster sandwich, the pseudo-inverse of the information, `fit_failure()` - the rule of a usable fit |
| `glmm_fit.py` | the GLMM by adaptive Gauss-Hermite quadrature and Newton steps on the marginal likelihood; `GlmmFitter` for repeated fits on one design; the threshold of a level |
| `load_spread.py` | the spread by the load: the loads of the requests, the groups of the axes, the rows `z`, the steps of the fit by the load |
| `variable_fit.py` | the fit of a quantity: the exclusion rule, the chain `VariableFitter`, the result `VariableFit` |
| `chain/` | the paths of the chain, the notes of its shunts and the numerical checks |
| `acceleration/` | the folded rows, the sparse design and the quadrature rule, built once |

## Key signatures

```python
fit_logit(rows: FoldedRows, run_weight, start=None, *, max_steps=55, tolerance=1e-8) -> LogitFit
fit_failure(converged: bool, steps: int, *estimates: np.ndarray) -> str | None   # None - a usable fit
GlmmFitter(design: SparseDesign, quadrature: QuadratureRule = GLMM_RULE)
GlmmFitter.fit(run_weight, theta_start, sigma2_start, modes_start=None) -> GlmmFit
GlmmFitter.fit_spread(run_weight, theta_start, gamma_start, spread_x, modes_start=None) -> SpreadFit
fit_glmm(rows, run_weight, theta_start, sigma2_start, modes_start=None) -> GlmmFit    # one fit
marginal_threshold(q: float, sigma2: float) -> tuple[float, float]    # eta* and d eta* / d sigma2
request_loads(points, layout) -> dict[str, RequestLoad]
spread_design(rows, loads) -> SpreadDesign    # .axes, .matrix, .grouped
fit_by_load(rows, weight, common: GlmmFit, loads) -> LoadFit | None    # one fit by the load
VariableFitter(paths=FIT_PATHS, checks=NUMERICAL_CHECKS, *, quadrature=GLMM_RULE, telemetry=None)
VariableFitter.fit(design: VariableDesign, run_weight, loads=None) -> VariableFit
fit_variable(design, run_weight, loads=None) -> VariableFit    # one fit along the default chain
```

## Setup and work

The quadrature rule of the marginal likelihood is built once, when `glmm_fit` is loaded (`GLMM_RULE`), and handed to every
`GlmmFitter`. For every quantity the chain folds the remaining rows once, builds their sparse design once and gives both to all
its fits: the plain logit, the GLMM from every start of the variance and the fit with the spread by the load. `fit_glmm`,
`fit_spread` and `fit_by_load` make one fit and build the sparse design for that call.

## Boundaries

The folder fits the folded matrices that `trials/design/` builds and gives the coefficients, their covariance and the notes of
the fit; it reads no files and computes no laws: the laws of the bins and the capacities come with `laws/` and `quantities/`.
The method names of a fit - `glmm_heterogeneous`, `glmm`, `logit`, `none` - are the vocabulary of the passport.
