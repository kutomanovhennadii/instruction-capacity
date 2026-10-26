# passport — the passport of a measurement from its published points

The package computes the VS-PMR passport of a published measurement: the capacities of the
model and everything they are computed from, and it verifies a published measurement.

## What is here

| Module | Role |
|---|---|
| `measured_points` | `MeasuredPoint` - a published record plus the trials the validator rebuilds from its answer |
| `analysis_area` | the purchase cube `D × m × n` from the start step `(4, 5, 1)` and its boundary; the points of ordinary cells |
| `bin_layout` | the bins of the passport axes and the doubling rule; the start step constants |
| `vpm_cube` | the run cube `V × P × M`: the cell of a point, the passport bins, the point weights |
| `passport_model` | the trial and the contract of a passport model: name, coefficients, design row, fit |
| `vspmr_model` | the public passport model `vspmr_glmm_v1`: parameter names, the design row of a trial, the rule price, the fit |
| `design` | the design matrix (a row per trial) and the exclusion rule of bins without an estimate |
| `logit_fit` | Newton on folded rows and the cluster sandwich covariance |
| `glmm_fit` | the GLMM fit: the shift of a point inside the likelihood; the threshold of a level |
| `model_fitting` | the result of a fit and the GLMM fit of the public model |
| `section` | the VS-PMR section: centred bins with their errors, excluded bins |
| `second_stage_laws` | the second-stage laws of the density, the number of rules and the rule number; the continuous IEC |
| `approximation` | the grid law by length slice in two steps, block shifts, restored bins, component correlations |
| `public_quantities` | the public quantities, one function each |
| `collapse` | the collapse of the answer: its rate over all answered points and its place over the words of the points that lost the text - bins, laws of the axes, shifts, final laws, checks |
| `passport_data` | the whole calculation: from the published records to the passport dictionary |
| `measurement_files` | the files of a measurement directory and of its passport |
| `verify` | the four checks of a published measurement |
| `document/` | `passport.md` and the figures of every model |

## Pipeline

1. **Trials.** For every `ok` record the validator parses the recorded answer and gives the
   trials (rule number, word position, outcome `z`) and the line where the model lost the
   text (a failure of the third kind).
2. **Area of analysis.** The purchase cube cuts `D`, `m`, `n` by doubling from the start step
   `(4, 5, 1)` up to the largest executed point (`ok` or `na`). A cell is represented by its
   upper corner. The state of the corner: three or more `api_error` points and no `ok` or
   `na` - `na`; fewer than three `ok`/`na` points - no state; no `ok` - `na`; every `ok`
   point lost the text - collapse; otherwise - open. A cell is a boundary cell when its
   corner collapsed or got `na`, unreachable when it has no state. The fit uses the points of
   the other, ordinary cells.
3. **Bins.** Text length `v = D·n` from 20 words by doubling: 1-20, 21-40, 41-80, ...;
   density - the levels `⌊log₂ p⌋` from `p = 1` down to the sparsest; the number of rules -
   `{1}, {2}, {3-4}, {5-8}, ...`; word position - the doubling bin of the answer line. The
   reference density bin (the implicit zero of `P`) is the bin of `⌈4/1⌉ = 4`.
4. **Weights.** The weight of a point is inverse to the filling of its run-cube cell,
   normalised to the mean weight 1; the effective number of points `n_eff = (Σw)²/Σw²` is
   published.
Steps 1-4 are computed once; steps 5-7 - for every passport model the passport is given,
through the contract `PassportModel`. The public passport has one model, `vspmr_glmm_v1`.

5. **Exclusion.** A bin where every replacement is right or every one wrong is removed with
   its trials (its coefficient has no finite maximum), until a fixed point. Removed bins are
   published with status `excluded` and no number.
6. **Fit with the shift of a point.** The plain logit (Newton on folded rows with point
   weights) gives the start; the GLMM fit puts the shift of a point `b ~ N(0, σ²)` inside the
   likelihood (adaptive Gauss-Hermite quadrature, Newton steps), and gives the coefficients,
   their covariance - the cluster sandwich, a cluster is a point - and `σ̂²` with its SE. If a
   fit does not converge, the passport is not built.
7. **Second stage and public quantities.** A law is drawn through the bins of every block by
   generalised least squares; the shifts go into the level `α_phys`, the bins are published
   restored; the public quantities are computed at `q = 0.95` and `0.5`, the level `q` read at
   the threshold `η*(q, σ²)`: the logit of the median point at which the share of right
   replacements over all points is `q`.
8. **Collapse of the answer.** The method of steps 5-7 applied to the outcome "lost the text":
   over all answered points, the boundary cells too, a point is one trial weighted by the run
   cube. The logit of the collapse is split into the bins of the text length and of the density
   (the weighted logit, the cluster sandwich by points); the length law
   `λ_Vψ + κ_ψ·log₂(v/20)` is drawn through the length bins with their inverse own variances as
   weights (they are strongly correlated through their reference bin, as the cells of the
   grid), the density law `μ_Pψ + A_ψ·p^γ_ψ` through the density bins over their full
   covariance; the shifts - the length law at `v = 20`, the density law at `p → 0` - go into the
   level `λ_ψ` of the rate law `logit ψ = λ_ψ + κ_ψ·log₂(v/20) + A_ψ·p^γ_ψ`. The bins of the
   number of rules are fitted once more as a check, with the Wald statistic "all of them are
   zero". The place of the collapse - the same method over the replacement words of the points
   that lost the text, the outcome of a word being "its line is not before the line where the
   answer lost the text": the axes word number `s` (the position bins of the grid), density and
   share of the text `(s − 1)/v` (ten equal bins); the word law is the plateau hyperbola of the
   passport with a rise past the corner `s_*` (a corner parameter on its bound is held), the
   density law `μ_PF + A_F·p^γ_F`, the share line `λ_QF + κ_QF·(s − 1)/v`, all over the full
   covariance; the shifts - the plateau, the density law at `p → 0`, the line at the start of
   the text - go into the level `λ_F`. The text length and the number of rules are checked.
   Computed once, not per model.
9. **Documents.** `passport_data.json`, `passport.md`, ten figures of every model in
   `figures/<model>/`, four figures of the rate `figures/collapse-*.png` and six of the place
   `figures/place-*.png`.

## Model

The logit of a right replacement:

```text
η = VS(v, s) + P(p) + M(m) + R(r) + b,   b ~ N(0, σ²)
```

`VS` - a free cell of the grid "length bin × answer position bin"; `P`, `M`, `R` - free bins of
the density `p = m/D`, the number of rules and the rule number; `b` - the shift of a point,
shared by all its trials and integrated out of the likelihood. The bins describe the median
point (`b = 0`). Second-stage laws: the grid - a hyperbola with the slanted asymptote
`λ_S + κ_S·log₂ s` and a plateau `μ(v) = μ₀ + (κ_V/2)·[log₂(v/v_*) + √(log₂²(v/v_*) + ω_V²)]` -
itself a hyperbola in the text length; the density - `μ_P − A_P·(1 − e^(−p/p_c))`; the number of
rules - `λ_M + κ_M·log₂ m`; the rule number - `λ_R − A_R·(r^γ − 1)`.

## Zeros

The grid law is fitted in two steps: every length slice gets its own plateau `μ_v` with shared
`κ_S`, `λ_S`, `ω_S`, then the plateau hyperbola `μ(v)` is drawn through the plateaus of all
determined slices (at least five). The grid law is shifted by its horizontal asymptote: at the
start of the answer (`s → 0`) of short texts (`v → 0`) the surface tends to `μ₀`, and
`δ_VS = μ₀`; `λ_S`, `μ_v` and the hyperbola are published minus `δ_VS`. A corner parameter `v_*`
or `ω_V` on a bound of its range is held: its value is the bound, it has no error, and the law
names it in `held`.

## Tolerance of `verify`

Reordering the summation of the same points moved the published passports of two
measurements by at most 1.5e-9 relative in the first stage, 3.5e-4 SE in the second-stage
numbers that have their own SE, and 3.2e-2 relative in the other second-stage numbers.
`verify` compares:

1. the composition exactly: keys, list lengths, strings, booleans, `null`, integers;
2. floats outside the `approximation` block, the `collapse` block too, within `1e-6` relative
   (plus `1e-12`);
3. numbers of the `approximation` block with their own SE in the same record within
   `0.01·SE`;
4. other floats of the `approximation` block within `0.1` relative (plus `1e-12`).

## Key signatures

```python
build_passport_data(description: MeasurementDescription, records: Sequence[PointRecord],
                    name: str, models: Sequence[PassportModel] = PUBLIC_MODELS) -> dict[str, Any]
fit_vspmr_glmm(design: Design, run_weights: Mapping[str, float] | None) -> FitResult
marginal_threshold(q: float, sigma2: float) -> tuple[float, float]   # η* and dη*/dσ²
verify_measurement(directory: Path, vocab_dir: Path) -> VerificationReport
compare_passports(published: Any, recomputed: Any) -> list[str]
```

## Errors

| Code | When |
|---|---|
| `PASSPORT_INPUT_MISSING` | the measurement has no points to fit (`PassportError`) |
| `FIT_NOT_CONVERGED` | the plain start or the GLMM fit did not converge (`PassportError`) |
| `PASSPORT_MODELS_INVALID` | the passport is given no model or a repeated model name |
| `MEASUREMENT_FILES_MISSING` | the measurement directory has no description or points file (`PassportError`) |
| `MODEL_FITTING_NON_FINITE` | the fit gave non-finite numbers or negative variances |
| `FIT_RUN_WEIGHTS_MISSING` | a point of the design has no weight |
| `RUN_CUBE_INVALID` | a node with `m > D` or `n < 1` |
| `BIN_LAYOUT_INVALID` | a bin number or value outside the layout |

## Boundaries

The package does not choose points and does not call the model: it reads published records.
There is no fallback fit when a fit does not converge. The public passport has one model; the
passport builds a section for every model it is given, through the contract.

Method: [The passport model](../../../docs/05-passport-model.md).
