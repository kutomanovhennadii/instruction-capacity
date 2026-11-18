# quantities — the sections of the three quantities

The sections of `P_sync`, `H` and `F` of the passport: the model of every quantity, its laws, level and capacities, assembled from
the common steps in `steps/`.

## What is here

| Module | Role |
|---|---|
| `sync_section` | the section of `P_sync`: model VS-PM, the grid with a plateau, the density and the number of rules, three capacities by the laws and the instruction capacity by the bins |
| `rule_word_section` | the section of `H`: model VS-PMR, the grid with the levels of its slices, the density, the number of rules and the rule number, the capacities by the laws and by the bins |
| `background_design` | the trials of `F` by the share of the answer and the folded matrix of model V-(s/v)-PM |
| `background_section` | the section of `F`: model V-(s/v)-PM, four laws through the well determined bins, three capacities by the laws and the instruction capacity by the bins |
| `steps/` | the common steps of a section |

## The steps of a section

Every section goes by the same steps in the same order: the fit with the variance by the load, checked against the median request
of the data (`steps/median_check`), and the approximations of the spread; the laws of the blocks, each chosen among the candidates
the assignment of forms gives it (`config/passport_laws.yaml`) by the smallest `QAIC`, with the weights `1/(σᵢ² + τ²)`
(`laws.block_choice`); the slices of a grid by the same rule (`laws.grid_slices`); the level - the sum of the shifts of the blocks;
the capacities by the laws and by the bins; the rows of the bins. The errors of every law are carried by the full covariance of
the fit, through the sensitivity of its parameters to the coefficients.

## Key signatures

    sync_section(points, layout, weights, assignment) -> dict
    sync_laws(design, fit, layout, support, assignment) -> dict
    rule_word_section(points, layout, weights, assignment) -> dict
    rule_word_laws(design, fit, support_words, assignment) -> dict
    background_cells(key, D, n, m, aligned, source_text, rule_table) -> BackgroundCells
    answer_share_design(points, layout, weights) -> AnswerShareDesign
    background_section(points, layout, weights, assignment) -> dict
    background_laws(built, fit, assignment) -> dict

## Boundaries

The sections compute; they write nothing. The passport is assembled from them by the root of the package, the document by
`document/`. The tests of the design of `F` live with the tests of its section: they share its fixtures.
