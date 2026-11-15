# charts — the kinds of figures behind one interface

The figures of the passport. A kind of figure draws one entry of the plan of a quantity: the bins of an axis with its law before and
after the shift correction, the cells of the grid by length slices with the curve of every slice, the levels of the slices with the
law of the level, the variance of the request shift by the groups of an axis with its approximation. The curve of a law is the
value of the form that the assignment of forms (`config/passport_laws.yaml`) gives its block; which figures a quantity gets, and in
which order, says the plan `config/passport_figures.yaml`.

## What is here

| Module | Role |
|---|---|
| `chart_frame` | the frame of a figure: the zero line, the abscissa, the grid, the legend, the title; the PNG written through a temporary file |
| `law_views` | the curves and the labels of the laws of the blocks by the forms of the collection; the registries of the views by quantity and block |
| `chart_kinds` | the interface `ChartKind`, the four kinds `axis`, `slices`, `plateau` and `spread`, the registry `CHART_KINDS` |
| `figure_plan` | the plan of the figures from its YAML file: the entries of every quantity with their kind, block and stem |
| `figure_writer` | the figures of a passport by the plan and the names of the figures of a quantity for the document |

## Figures

Every figure of an axis shows the bins joined by a line and the law as the red curve, before the correction (the coefficients of the
fit) and after it (minus the shift of the axis). The whiskers of an axis are the own error of a bin, ±1.96·√q; the whiskers of a grid
cell are ±1.96·SE. An open point is the reference bin, a diamond is a bin outside the law.

| Folder | Files |
|---|---|
| `figures/P_sync/` | `S-*`, `plateau-before-correction`, `P-*`, `M-*` |
| `figures/H/` | `S-*`, `plateau-before-correction`, `P-*`, `M-*`, `R-*` |
| `figures/F/` | `V-*`, `answer-share-*`, `P-*`, `M-*` |

A quantity fitted by the GLMM with heterogeneous random-intercept variance also has `spread-V`, `spread-P` and `spread-M`: the
variance of the shift by the groups of an axis with the other two axes averaged over the requests, the whiskers ±1.96·SE, and the
approximation through the groups by its published form: `a + b·log₂ x`, `c + A/(1 + x/x₀)` or `c + A/x`. `*` stands for
`before-correction` and `after-correction`. A quantity without laws gets no figures.

## Key signatures

    load_figure_plan(path) -> FigurePlan
    figure_names(plan, quantity, section) -> list[str]
    write_passport_figures(passport, out_dir, assignment, plan) -> tuple[Path, ...]

## Errors

| Code | When |
|---|---|
| `FIGURE_PLAN_CONFIG_INVALID` | the plan file is missing, is not YAML or breaks the schema (`ConfigError`) |
| `FIGURE_VIEW_UNKNOWN` | the plan names a block of a quantity that no view draws (`ContractViolationError`) |

## Boundaries

The figures fit nothing: every number comes from the passport dictionary, and the curves have no formulas of their own. A new figure
of an existing kind is a line of the plan; a new kind is a class behind `ChartKind` and a line of `CHART_KINDS`.
