# chain — the chain of fits of a quantity

The fit of a quantity goes along a chain of paths: the plain logit, the GLMM with one variance of the request shift and the
GLMM with heterogeneous random-intercept variance, where the variance depends on the load. Every path builds on the fit before
it; a path that gives no fit stops the chain, and the quantity keeps the fit reached so far. Such a fallback is a shunt - the
rule of a non-standard situation (specification §12.8). The estimates of the reached path then go through the numerical checks.

## What is here

| Module | Role |
|---|---|
| `fit_paths.py` | the paths `LogitPath`, `GlmmPath`, `LoadSpreadPath` behind one interface, the default chain `FIT_PATHS`, `check_chain()`, the method names and `SIGMA2_STARTS` |
| `numerical_checks.py` | the checks `FiniteNumbersCheck`, `NegativeVarianceCheck`, `InformationCheck` behind one interface, `NUMERICAL_CHECKS`, `check_fit()` |
| `fit_notes.py` | `FitNote` and the codes of the catalog of shunts |

## Key signatures

```python
FitPath.advance(problem: FitProblem, state: ChainState, telemetry: TelemetryHub) -> PathStep
FitPath.estimate(problem: FitProblem, state: ChainState) -> PathEstimate
check_chain(paths: Sequence[FitPath]) -> None    # ConfigError(FIT_PATHS_INVALID) on a wrong order
NumericalCheck.inspect(evidence: FitEvidence) -> tuple[FitNote, ...]
check_fit(evidence: FitEvidence, checks=NUMERICAL_CHECKS) -> tuple[FitNote, ...]
FitNote(code: str, message: str, level: int = logging.WARNING)
```

## The catalog of shunts

Every place answers the three questions of the specification: is there a shunt, is it one rule for a whole class of situations,
and is it seen in the log and the telemetry. Every row below is one rule for every quantity and every measurement.

| Place | Shunt | Log | Telemetry |
|---|---|---|---|
| no rows or columns remain after the exclusion rule | method `none` | `WARNING FIT_NO_DATA` | event `fit_variable` |
| the plain logit does not converge or gives non-finite estimates | method `none` | `WARNING LOGIT_FAILED` | event `fit_logit` |
| the GLMM from a start of the variance gives no usable fit | the next start of `SIGMA2_STARTS` | `INFO GLMM_START_FAILED` | an event `fit_glmm` per start |
| the GLMM gives no usable fit from any start | method `logit` | `WARNING GLMM_FAILED` | events `fit_glmm` |
| the variance of the shift stays on its floor | method `glmm`, the spread is not fitted | `INFO SPREAD_ON_FLOOR` | no event |
| every axis of the spread has one group | method `glmm` | `INFO SPREAD_ONE_GROUP` | no event |
| the fit with the spread by the load gives no usable fit | method `glmm` | `WARNING SPREAD_FAILED` | event `fit_spread` |
| a group of the spread stops on the lower bound `ln σ = ln 10⁻⁴` | the group is merged with its neighbour on the axis and the spread is fitted again | `INFO SPREAD_MERGED` | no event |
| the matrix of the second derivatives (the total derivative, the nodes of the quadrature moving with the parameters) is not negative definite at a Newton step | the step takes the moduli of its eigenvalues | the end state is checked by `InformationCheck` | no event |
| a Newton step lowers the likelihood | the step is halved up to 30 times; then the fit stops unconverged | through the note of its path | no event |

The numerical checks are not shunts: they report and change nothing.

| Check | Finding | Log |
|---|---|---|
| `FiniteNumbersCheck` | a number of the estimates, their covariance or the information is not finite | `WARNING NUMBERS_NOT_FINITE` |
| `NegativeVarianceCheck` | a variance of the estimates is negative; its standard error is taken as zero | `WARNING VARIANCE_NEGATIVE` |
| `InformationCheck` | directions of the information not above `INFORMATION_RCOND` of its largest eigenvalue get zero variance | `WARNING INFORMATION_UNSUPPORTED` |
| `InformationCheck` | the information has negative eigenvalues: the estimates are not at a maximum | `WARNING INFORMATION_INDEFINITE` |

A line of the log reads `<quantity>: <message> [<code>]`; the notes stay in `VariableFit.notes` in their order.

## Boundaries

A new path is a class of the interface and a place in the chain; `check_chain` rejects a path that comes before the fit it
builds on. A new check is a class of its interface and an entry of `NUMERICAL_CHECKS`. A shunt or a check that would change the
numbers of a passport is a decision about the passport, not a change of this folder.
