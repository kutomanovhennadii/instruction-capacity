# document — `passport.md` and the figures of every quantity

Renders the passport dictionary (`passport_data.json`) into a human-readable document and its figures: 24, and three more for
every quantity with the spread by the load. The document is in English only. Every number comes from the passport dictionary;
nothing is fitted or recomputed here.

## What is here

| Module or folder | Role |
|---|---|
| `renderer` | assembles `passport.md` from the parts of the quantities and writes the figures and the document, each through a temporary file |
| `sections/` | the renderer of every quantity: `sync_text`, `rule_word_text`, `background_text` |
| `text_cells` | the cells of the text shared by the quantities: the sentence about the fit, the table of the capacities with the texts of their statuses, the table of the spread, the links to the figures |
| `number_text` | the numbers of the text and of the labels of the figures: a true minus sign and the precision of a capacity |
| `law_drawings` | the forms of the approximations of the spread, shared by the text and the figures |
| `charts/` | the kinds of figures, the curves of the laws by their forms and the plan of the figures (`config/passport_figures.yaml`) |

## Sections of `passport.md`, in order

1. **The formula** - `P_correct = P_sync · [π · H + (1 − π) · (1 − F)]` and the zero point of a capacity; the level 0.5 of the
   capacities.
2. **P_sync** - the fit, the variance of the shift by the groups of the axes, the laws of the grid, of the density and of the number
   of rules, the level, the capacities, the figures.
3. **H** - the same with the law of the rule number.
4. **F** - the same with the laws of the length, of the share of the answer, of the density and of the number of rules.

Every law is written with the form the data chose by the smallest `QAIC`, its formula, parameters and fit; the law of the grid
names the form of the slices, its shared parameters and the own level `μ` of every slice (`text_cells.slice_sentence`).
A capacity is written with its interval; a value beyond the measured area is marked as a continuation of the law. A capacity or a
bound that the passport holds as `null` because a floating-point number does not hold it is written as more than 10³⁰⁸. Capacities
by the bins stand next to those by the law. What a status of a capacity says is a table of `text_cells`, not a chain of branches.

## Key signatures

    write_documents(passport, directory, assignment, plan) -> tuple[Path, tuple[Path, ...]]
    render_passport(passport, plan) -> str

## Errors

| Code | When |
|---|---|
| `SPREAD_FORM_UNKNOWN` | an approximation of the spread has a form that no drawing knows (`ContractViolationError`) |

## Boundaries

The document computes nothing and fits nothing. The names of the figures of a quantity come from the plan of the figures, the same
plan that draws them, so the document links exactly the files that the figures write.

Method: [The passport model](../../../../docs/05-passport-model.md).
