# document — `passport.md` and the figures of every model

Renders the passport dictionary (`passport_data.json`) into a human-readable document,
ten figures per model of the passport and ten figures of the collapse of the answer. The
document is in English only. Every number comes
from the passport dictionary; nothing is recomputed.

## Sections of `passport.md`, in order

Sections 2-5 belong to the models: each has its heading once and a subsection
``### `<model>` `` for every model of the passport, in its order. The public passport has one
model.

1. **Totals** - published and answered points, provider tokens, money (tokens × the tariff of
   the measurement), Bernoulli trials, the note on incomplete provider counters, the alarms.
2. **Capacities and loads** - the public quantities with `± 1.96·SE`.
3. **Figures** - links to the figures of the model.
4. **Model parameters** - the formula of the model (with the shift of a point `b` when it is
   inside the fit) and the second-stage laws in LaTeX (`$$…$$`), `α_phys`, the block shifts and
   the law parameters.
5. **Fit diagnostics** - the between-point variance with its origin (`inside the fit` for
   `vspmr_glmm_v1`), the weighting, the plateau by length slice, the restored bins, the
   canonical correlation of the components.
6. **Collapse of the answer** - how many answered points lost the text; for the rate and for the
   place: the formulas, the final law and the shifts with their parameters, the bins, the checks
   with their Wald statistics, the figures; absent without the `collapse` block.
7. **Bin cube filling** - the purchase cube as text.

## Figures

The figures of a model are written into `figures/<model>/`. Every figure comes in two states:
before the shift of its block - the raw coefficients of the bins, the coding of the fit - and
after it - minus the shift, the law at the zero of the axis moved into `α_phys`. The whiskers
are ±1.96 own SE of a point; the error of the shift is common to all points of a block and is
not drawn. A model without the shifts of its blocks (not all laws are measured) has no figures.

| File | What it shows |
|---|---|
| `vs-slices-before-shift.png`, `vs-slices-after-shift.png` | the grid cells of all length slices on one image and the slice curves with shared `κ_S`, `ω_S` and own plateau `μ_v` |
| `plateau-before-shift.png`, `plateau-after-shift.png` | the plateaus `μ_v` by length bin and the plateau hyperbola `μ(v)` through them |
| `P-before-shift.png`, `P-after-shift.png` | the density bins and their law |
| `M-before-shift.png`, `M-after-shift.png` | the rule-count bins and their law |
| `R-before-shift.png`, `R-after-shift.png` | the rule-number bins and their law |

The collapse of the answer belongs to the whole passport, and its figures lie in `figures/`
itself, drawn the same way as the axes of a model, before and after the shift of the axis:

| File | What it shows |
|---|---|
| `collapse-V-before-shift.png`, `collapse-V-after-shift.png` | the text-length bins of the logit of the collapse and the length law |
| `collapse-P-before-shift.png`, `collapse-P-after-shift.png` | the density bins of the logit of the collapse and the density law |
| `place-S-before-shift.png`, `place-S-after-shift.png` | the word-number bins of the logit of the place and the word law |
| `place-P-before-shift.png`, `place-P-after-shift.png` | the density bins of the logit of the place and the density law |
| `place-Q-before-shift.png`, `place-Q-after-shift.png` | the bins of the share of the text and the share line, on a linear abscissa |

A passport without the shifts of the rate (of the place) has no rate (place) figures.

## Modules

- `labels.py` - section titles and labels of the public quantities;
- `formatting.py` - number formatting;
- `sections.py` - one function per section;
- `figures.py` - the figures of every model and the collapse figure;
- `renderer.py` - `render_passport(passport, figures)` and `write_documents(passport, directory)`.

Method: [The passport model](../../../../docs/05-passport-model.md).
