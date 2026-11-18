# laws — the laws through the fitted bins

The second stage of the passport: smooth laws through the fitted bins of a block, the shifts of the blocks and the tools that turn
laws into rows of the passport. The forms of the laws live in `forms/`; the laws of a quantity and its model live in
its section (`../quantities/`).

## What is here

| Module | Role |
|---|---|
| `gls_tools` | generalised least squares through the bins: the points of a series, the solver on whitened residuals, the sensitivity of the estimates to the coefficients, the delta method; the statuses of a published quantity |
| `law_rows` | the joint covariance of a fit, half-widths, own errors of bins (quasi-variances), the facts of a fit with the spread by the load, the row of a capacity by a law; a number of the row past the largest floating-point number, about `1.8·10³⁰⁸`, is `None` |
| `spread_laws` | the approximations of the variance of the request shift by the groups of an axis, with the forms from the assignment |
| `grid_slices` | the law of the length slices of a grid: a form of the collection whose level `μ` every slice owns while the other parameters are shared (`SlicedForm`), the choice of that form by the smallest `QAIC` and its entry in the passport |
| `plateau_grid` | the law of the length × word-number grid of `P_sync`: the law of the slices by `grid_slices`, then the law of the plateau through the levels of the slices chosen by `block_choice` |
| `block_choice` | the law of a block: the form with the smallest `QAIC` among the candidates of the block, the outliers clipped, the weights `1/(σᵢ² + τ²)`; the shift of a block, the rows of a law over the coefficients (`BlockLaw`, `LawStack`) and its entry in the passport |
| `point_weights` | the weighting of any subset of the bins of a law: the own variances or none (`Weigh`); the scatter `τ²` of the points beyond their own variances by DerSimonian and Laird |
| `sigma_clipping` | the outlier clipping of a law by the algorithm of astropy: one scale for all the bins, three scales from the median of the residuals, refit until the number of the clipped bins stops changing |
| `form_threshold` | the choice among the forms of a block: the smallest `QAIC = χ²/ĉ + 2k` wins (Burnham and Anderson 2002), the established form takes a tie, an alternative with a parameter held on a bound does not replace it |
| `forms/` | the collection of the forms of laws, their fit and the assignment of the forms to every block (`config/passport_laws.yaml`: the established form first, then the alternatives) |

## Key signatures

    named_points(theta, covariance, names, prefix, abscissa, abscissae=None) -> (x, values, block, index, names)
    gls_least_squares(curve, jacobian, start, lower, upper, x, beta, root, method="trf") -> (result, chi2)
    joint_covariance(fit) -> ndarray; half_width(gradient, joint) -> float
    own_variances(block, implicit_reference) -> ndarray; fit_summary(fit) -> dict
    capacity_row(key, level, log_value, half, support, scale=1.0) -> dict
    power_of_two(exponent, scale=1.0) -> float | None; exponential(exponent) -> float | None
    spread_forms(assignment, quantity) -> dict; spread_laws(spread, forms) -> dict
    choose_slice_form(forms, x, values, variances, count) -> SliceLaw | reason
    slice_entry(law, rows, joint, symbol) -> dict; slice_value(form, shared, level, ell) -> ndarray
    fit_vs_law(theta, covariance, names, abscissae, layout, slice_forms, plateau_forms) -> dict
    vs_curve(slice_form, shared, plateau_form, plateau_params, xu) -> ndarray
    choose_law(forms, x, values, weigh, *, starts=None) -> BlockChoice | reason
    choose_scattered_law(forms, x, values, variances, *, starts=None) -> BlockChoice | reason
    block_law(choice, point_rows, bins, x, at_infinity) -> BlockLaw   # .kept_bins, .outliers, .shift()
    law_entry(law, joint, symbol) -> dict   # law_form, established_form, c_hat, outliers, n_points, ...
    variance_weights(variances) | plain_weights() -> Weigh
    dersimonian_laird(jacobian, chi2, variances) -> float
    clipped_fit(form, x, values, weigh, *, starts=None) -> ClippedFit | None
    pick(contenders, points) -> (position, c_hat)

## The law of a block

Every law of the passport goes through every bin of its block with a coefficient: a bin leaves a law only as an outlier. A
bin weighs `1/(σᵢ² + τ²)`: `σᵢ²` is its own variance, `τ²` the scatter of the kept bins beyond their own variances around the
converged form with the smallest `χ²` per degree of freedom, by the estimate of DerSimonian and Laird for a regression with the
derivatives of the law in place of its design; so the curvature of a law does not count as scatter. The forms of the block are
chosen first with the weights of the own variances, which gives the kept bins and `τ²`, and then again with `1/(σᵢ² + τ²)`. The
established form of the block is fitted with the clipping first; every form of the block is then fitted through the bins it
kept with the same weights, and the form with the smallest `QAIC` is taken, the established form winning a tie. `ĉ` is the
smallest `χ²` per degree of freedom among the converged forms, not below one: the bins scatter around any law more than their
own errors allow, and a flexible form that lies badly does not inflate it. An alternative with a parameter held on a bound of
its range does not replace the established form. A form without a finite value at a bin does not converge and drops out. When
no form converges the line is the law; when the line does not converge either, the block has no law and the reason is written.

The slices of a grid choose their form in the same way (`grid_slices`): all the cells go through one fit, the curve shared by
the slices and the level `μ` of every slice its own.

## Boundaries

The folder carries no formula of a model: which blocks a quantity has, how its shifts make its level and how its capacities read
its laws stay in the section of the quantity (specification §2.8). The forms of the slices and of the plateau come from the
assignment of forms, as the forms of every block. The approximations of the spread publish the names `line`, `hyperbola` and
`inverse`, as the passport has them.
