# steps — the common steps of a section

The steps the sections of the three quantities share; a section composes them with the formulas of its own model.

## What is here

| Module | Role |
|---|---|
| `axis_rows` | the rows of the bins of an axis of `P_sync` and `H`: raw, minus the shift, own errors, the law, the counts; the status `law`, `outlier` (clipped from the law), `reference` or `excluded` |
| `cell_rows` | the rows of the cells of the grid of `P_sync` and `H`: raw, minus the shift of the grid, the law the section gives |
| `bins_walk` | the instruction capacity by the bins of the number of rules for `P_sync` and `F`: the walk that spends the reserve |
| `law_capacity` | the capacity by the root of laws of any form, with the status `no_slope` for laws without a root; the measure of the density `density_odds_factor = e^\|P̂(1)\|`; a factor or a bound past the largest floating-point number is `None` |
| `section_fit` | the first stage of a section, `SectionFit`: the design of a quantity without the rows of its fit (`light_design`), the fit and the longest text of the measurement |
| `median_check` | the fit of a quantity with the variance by the load, checked against the median request of the data on the axes constant within a request; a bin turned to the other side of the reference bin beyond the error of the comparison sends the quantity to one variance with the note `SPREAD_AGAINST_DATA` |

## Key signatures

    axis_rows(design, fit, letter, key, in_law, law_at, shift, shift_row, joint) -> list[dict]
    cell_rows(design, fit, shift, shift_row, joint, law_at) -> list[dict]
    capacity_by_bin_walk(rows, delta_m, reserve) -> dict
    root_capacity_row(key, function, stack, joint, support, *, scale=1.0, ceiling=inf) -> dict
    requests_of(points, layout, outcome) -> list[Request]; axis_bins(design, letters) -> dict[str, AxisBins]
    median_comparison(requests, axis, reference) -> dict[int, float]
    spread_against_data(requests, axes, theta) -> list[str]
    checked_fit(design, run_weight, loads, requests, axes) -> VariableFit

## The check against the median request

A bin of the passport is the logit of the median request of the model; where the spread of the requests changes along an axis,
the bin can lie on the other side of the reference bin than the requests themselves. For every bin of the density, the number of
rules and, for `F`, the text length, the requests with at least 50 trials are grouped into the cells of the other two axes; in
every cell the median shares of failures of the requests of the bin and of the reference bin go to the logit of the odds of a
success, and their difference is averaged over the cells with the weight of the number of the requests of the reference bin. Its
error comes from 200 resamplings of the requests with the seed 0. A bin and its comparison on the two sides of the reference bin,
the comparison beyond the half-width of its 95% interval, send the quantity to the fit with one variance.

## Boundaries

A step carries no formula of a model: the law of the grid at a cell comes from the section. `H` prices every rule along both axes of the rules, so its capacity by the bins stays in its
section; `F` has its own design of the bins and its own rows.
