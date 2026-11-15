# steps — the common steps of a section

The steps the sections of the three quantities share; a section composes them with the formulas of its own model.

## What is here

| Module | Role |
|---|---|
| `axis_rows` | the rows of the bins of an axis of `P_sync` and `H`: raw, minus the shift, own errors, the law, the counts; the status `law`, `outlier` (clipped from the law), `reference` or `excluded` |
| `cell_rows` | the rows of the cells of the grid of `P_sync` and `H`: raw, minus the shift of the grid, the law the section gives |
| `bins_walk` | the instruction capacity by the bins of the number of rules for `P_sync` and `F`: the walk that spends the reserve |
| `law_capacity` | the capacity by the root of laws of any form, with the status `no_slope` for laws without a root; the measure of the density `density_odds_factor = e^\|P̂(1)\|` |

## Key signatures

    axis_rows(design, fit, letter, key, in_law, law_at, shift, shift_row, joint) -> list[dict]
    cell_rows(design, fit, shift, shift_row, joint, law_at) -> list[dict]
    capacity_by_bin_walk(rows, delta_m, reserve) -> dict
    root_capacity_row(key, function, stack, joint, support, *, scale=1.0, ceiling=inf) -> dict

## Boundaries

A step carries no formula of a model: the law of the grid at a cell comes from the section. `H` prices every rule along both axes of the rules, so its capacity by the bins stays in its
section; `F` has its own design of the bins and its own rows.
