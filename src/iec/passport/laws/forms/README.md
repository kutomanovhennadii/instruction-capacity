# forms — the collection of the forms of laws

A law of the passport is a smooth curve of a few parameters through the fitted bins of a block. A form is a family of such
curves behind one interface: the value, the derivatives by the parameters for the delta method, the limit at `x → −∞`, the
formula for the document, the names of the parameters, the range and the starts of a fit and the parameters that the data may
push to a bound. A form knows nothing of a quantity or a block: the block gives it the abscissa, and the assignment says which
form the block takes.

## What is here

| Module | Role |
|---|---|
| `law_form.py` | the protocol `LawForm`, `law_shift()` - the shift of a law at the zero of its axis, `unbounded_range()`, the error codes |
| `linear_forms.py` | `LinearForm` `a + b·x`, `LogarithmicForm` `a + b·log₂(x/x₀)`, `LogLinearForm` `a + b·log₂(x/x₀) + c·x` |
| `hyperbolic_forms.py` | `HyperbolicForm` `c + A/x`, `FractionalHyperbolicForm` `c + A/(1 + x/x₀)`, `SlantedHyperbolicForm` `μ + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]`, `TwoSlopeHyperbolicForm` `μ + β·(x − x*) + (κ/2)·[(x − x*) + √((x − x*)² + ω²)]` - two slanted asymptotes, the slopes `β` and `β + κ` of any sign, no horizontal asymptote, `AsymptoticHyperbolicForm` - the slanted hyperbola written by its slanted asymptote `κx + λ` |
| `growth_forms.py` | `ExponentialForm` `c + A·e^(−x/τ)`, `BinaryExponentialForm` `μ − A·2^(γx)`, `SaturationForm` `μ − A·(1 − e^(−x/x_c))`, `PowerForm` `λ − A·(x^γ − 1)` |
| `law_fit.py` | `fit_law()` - the fit of a form through points over their weight matrix; the weightings `covariance_weighting()`, `variance_weighting()`, `plain_weighting()`; the result `LawFit` |
| `form_registry.py` | `LAW_FORMS` - every form by its name and the options it takes; `build_form()` |
| `law_assignment.py` | the assignment file `config/passport_laws.yaml`: its model, `load_law_assignment_config()`, `build_law_assignment()`, `LawAssignment` |

## Key signatures

```python
LawForm.value(params, x) -> np.ndarray
LawForm.jacobian(params, x) -> np.ndarray    # (points, parameters)
LawForm.limit(params) -> float | None        # the horizontal asymptote at x -> -inf
LawForm.formula(symbol: str) -> str
law_shift(form, params, zero: float) -> float    # zero = -inf: the limit, else ContractViolationError(LAW_SHIFT_UNDEFINED)
fit_law(form, x, values, weighting, *, starts=None, bounds=None) -> LawFit
covariance_weighting(block) -> LawWeighting    # P = pinv(block), its root L with P = L·Lᵀ
build_form(name: str, options=None) -> LawForm    # ConfigError: LAW_FORM_UNKNOWN, LAW_FORM_OPTIONS_INVALID, LAW_FORM_INVALID
build_law_assignment(load_law_assignment_config(path)) -> LawAssignment
LawAssignment.form_of(quantity, block) -> LawForm    # ContractViolationError(LAW_BLOCK_UNKNOWN)
LawAssignment.spread_form_of(quantity, axis) -> LawForm
```

## The fit

The residuals `Lᵀ·(β − f(x))` are whitened by the root of the weight matrix; the starts are tried in order, and the smallest
`χ²` wins. A parameter that the data push to a bound of its range is held there: its value is the bound, and its row of the
sensitivity `(JᵀPJ)⁻¹JᵀP` is zero. A form with such parameters is fitted by `dogbox`, which stops on a bound, the others by
`trf`. Singular normal equations - a law that its points do not determine - give a fit without convergence. With the covariance
weighting and the default starts of the slanted hyperbola this is the procedure of the density law of P_sync.

## The abscissa and the formulas

The block gives the abscissa: the load, its logarithm or its logarithm against a reference. `np.log2` of an array and
`math.log2` of a number differ in the last bit for about two values in a hundred, so a block computes its abscissa the way the
passport computes it. The formulas of the forms that the passport uses repeat its expressions operation by operation; on the
measurement v55 the forms and the fit give the same values, derivatives, density law and shifts as the passport code, bit for
bit.

## Boundaries

A new form is a class of its module and a line of `LAW_FORMS`; neither the fit nor the assignment changes. The folder holds the
forms and their fit; the laws of the blocks with their own recipes of a fit and the rows of the passport live in `laws/`, the
figures and the document in `document/`.
