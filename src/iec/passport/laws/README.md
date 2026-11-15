# laws — the laws through the fitted bins

The second stage of the passport: smooth laws through the fitted bins of a block, the shifts of the blocks and the tools that turn
laws into rows of the passport. The forms of the laws live in `forms/`; the laws of a quantity and its model live in
its section (`../quantities/`).

## What is here

| Module | Role |
|---|---|
| `gls_tools` | generalised least squares through the bins: the points of a series, the solver on whitened residuals, the sensitivity of the estimates to the coefficients, the delta method; the statuses of a published quantity |
| `law_rows` | the joint covariance of a fit, half-widths, own errors of bins (quasi-variances), the facts of a fit with the spread by the load, the row of a capacity by a law |
| `spread_laws` | the approximations of the variance of the request shift by the groups of an axis, with the forms from the assignment |
| `plateau_grid` | the law of the length × word-number grid with a plateau: a hyperbola by length slice, then the law of the plateaus chosen by `block_choice` |
| `block_choice` | the law of a block: the established form of the block unless an alternative is better beyond the threshold, the outliers clipped; the shift of a block, the rows of a law over the coefficients (`BlockLaw`, `LawStack`) and its entry in the passport |
| `point_weights` | the weighting of any subset of the bins of a law: the full covariance, the own variances or none (`Weigh`) |
| `sigma_clipping` | the outlier clipping of a law by the algorithm of astropy: one scale for all the bins, three scales from the median of the residuals, refit until the number of the clipped bins stops changing |
| `form_threshold` | the threshold between the established form and its alternatives: `QAIC = χ²/ĉ + 2k`, an alternative replaces the established form when it is better by more than 10 (Burnham and Anderson 2002) |
| `forms/` | the collection of the forms of laws, their fit and the assignment of the forms to every block (`config/passport_laws.yaml`: the established form first, then the alternatives) |

## Key signatures

    named_points(theta, covariance, names, prefix, abscissa, abscissae=None) -> (x, values, block, index, names)
    gls_least_squares(curve, jacobian, start, lower, upper, x, beta, root, method="trf") -> (result, chi2)
    joint_covariance(fit) -> ndarray; half_width(gradient, joint) -> float
    own_variances(block, implicit_reference) -> ndarray; fit_summary(fit) -> dict
    capacity_row(key, level, log_value, half, support, scale=1.0) -> dict
    spread_forms(assignment, quantity) -> dict; spread_laws(spread, forms) -> dict
    fit_vs_law(theta, covariance, names, abscissae, layout, reference_total_words) -> dict
    choose_law(forms, x, values, weigh, *, starts=None) -> BlockChoice | reason
    block_law(choice, point_rows, bins, x, at_infinity) -> BlockLaw   # .kept_bins, .outliers, .shift()
    law_entry(law, joint, symbol) -> dict   # law_form, established_form, c_hat, outliers, n_points, ...
    covariance_weights(block) | variance_weights(variances) | plain_weights() -> Weigh
    clipped_fit(form, x, values, weigh, *, starts=None) -> ClippedFit | None
    pick(contenders, points, established) -> (position, c_hat)

## The law of a block

Every law of the passport goes through every bin of its block with a coefficient: a bin leaves a law only as an outlier. The
established form of the block is fitted with the clipping first; every form of the block is then fitted through the bins it
kept, and an alternative replaces the established form only when its `QAIC` is lower by more than 10. `ĉ` is `χ²` per degree
of freedom of the most flexible converged form, not below one: the bins scatter around any law more than their own errors
allow. A form without a finite value at a bin - the hyperbola at `x = 0` - does not converge and drops out. When no form
converges the line is the law; when the line does not converge either, the block has no law and the reason is written.

## Boundaries

The folder carries no formula of a model: which blocks a quantity has, how its shifts make its level and how its capacities read
its laws stay in the section of the quantity (specification §2.8). The grid with a plateau keeps the joint fit of its slices here;
its plateau law is the slanted hyperbola of the collection. The approximations of the spread publish the names `line`,
`hyperbola` and `inverse`, as the passport has them.
