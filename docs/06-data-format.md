# Data format

A published measurement is a directory `artefacts/<measurement>/`:

| File | Content |
|---|---|
| `measurement.json` | the description of the measurement |
| `points.jsonl.gz` | the records of the points, gzip-compressed JSON lines. Points that would make the file larger than 100 MiB, the limit of a file on GitHub, are kept in numbered parts instead, `points-001.jsonl.gz`, `points-002.jsonl.gz` and so on, read one after another in the order of their numbers |
| `passport_data.json` | the passport: the fits, bins, laws, levels and capacities of the three quantities |
| `passport.md`, `figures/` | the passport as a document, with the figures of every quantity in `figures/P_sync/`, `figures/H/` and `figures/F/` |
| `passport.pdf` | the same document as a PDF |

The data are what was sent to the model and what it answered, plus the numbers that rebuild the
task. What the code computes from them (the prompt, the reference answer, the alignment, the trials)
is not stored.

## Point record

One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
| `format_version` | integer | the version of the format of the record; `1`. A record without the field is of version 1; a newer version is refused |
| `D`, `n`, `m` | integer | conditions: distinct words, occurrences of each, rules |
| `seed_rules`, `seed_text`, `seed_order` | integer | seeds of the rule list, of the text and of the line order ([task construction](03-task-construction.md)) |
| `rules` | array of `[word, replacement]` | the rule list in the order shown to the model |
| `source_text` | string | the text as sent: words separated by spaces, lines of ten words separated by `\n` |
| `answer` | string or `null` | the raw answer; `null` if the model was not called or did not answer |
| `status` | string | `ok`, `na` or `api_error` |
| `prompt_tokens`, `response_tokens` | integer or `null` | the token counters reported by the provider; `null` if it did not report them |

The six numbers `D, n, m, seed_rules, seed_text, seed_order` identify a point within a
measurement. Seeds are integers of arbitrary size (up to about $10^{44}$): read them as integers,
never as floating point.

`seed_order = 0` means that the rule lines stand in the order of the draw; any other value is the
seed of a permutation of the same pairs.

**Order and uniqueness.** A published measurement holds one record per point, the last record of
that point, sorted by `D, m, n, seed_rules, seed_text, seed_order`. The order of measurement is not
published. A point whose call failed to the end is published with the status `api_error`.

## Measurement description

| Field | Meaning |
|---|---|
| `format_version` | the version of the format of the description; `1`. A description without the field is of version 1; a newer version is refused |
| `model` | `model_id` (as named by the provider), `provider`, `gen_params` (`temperature`, `max_tokens` = the answer reserve, `top_p`, `seed`), `tokenizer_name`, `context_window_tokens`, `response_ceiling_tokens` |
| `protocol_version` | the version of the protocol: prompt template, text layout, validation rules; `"0.2"`. Version 0.2 keeps the prompt and the text of version 0.1 and reads an answer by the alignment of places |
| `vocab_id`, `vocab_hash` | the vocabulary and its hash |
| `tariff` | `input_usd_per_million`, `output_usd_per_million`: the price the points were paid at; `null` if the money is not computed |
| `files` | file name → SHA-256 of every data file: `points.jsonl.gz` or each of its numbered parts |

The passport and the PDF are not listed: `verify` recomputes the passport, and a PDF is not
comparable byte for byte.

## Passport data

The top keys of `passport_data.json`:

| Key | Meaning |
|---|---|
| `model` | `id`, the model id; `experiment_id`, the name of the measurement |
| `points` | the number of answered points |
| `formula` | the formula that combines the three quantities |
| `weighting` | the weights of the run cube: `cells`, `runs`, `max_occupancy`, `n_eff`, `min_weight`, `max_weight` |
| `P_sync`, `H`, `F` | the sections of the three quantities |

A section of a quantity:

| Key | Meaning |
|---|---|
| `model`, `weights` | the name of the model and of the cube of the weights |
| `fit` | `method` (`glmm_heterogeneous` for the GLMM with heterogeneous random-intercept variance, `glmm`, `logit` or `none`), `sigma2` (for `glmm_heterogeneous` the mean over the requests), `sigma2_half`, `points`, `rows`, `trials`, `failures`, `parameters`, `steps`, `loglik`, `dropped` (the removed bins), `spread` (for `glmm_heterogeneous`, below) |
| `spread_laws` | for `glmm_heterogeneous`: the approximations of the variance by the groups of the axes, below |
| `status` | `measured` if the quantity has laws; otherwise it says why it has none |
| `bins` | the rows of the axes (`grid`, `slices`, `P`, `M`, `R`, `V`, `q`, by the model) |
| `laws` | the parameters of the laws, with `half` for the half-widths of their 95% intervals and `held` for the parameters on a bound of their range; a law of a block has `law_form` (its form), `established_form` (the form of the block it replaced or kept), `c_hat` (the overdispersion of the criterion of the choice), `candidates` (every form with `converged`, `chi2`, `free` and `criterion`, the QAIC), `outliers` (the bins clipped from the law) and `n_points` (the bins it goes through) |
| `shifts`, `alpha` | the shifts of the blocks and the level, each with `value` and `half` |
| `capacities` | the capacity rows |
| `density_loss_at_one` | for `H`: the loss of the density at `p = 1` |
| `edge_rules`, `edge_words` | for `F`: the edge of the measured area |

A row of an axis has `name`, `center` (the centre of mass of the bin), `raw` (the coefficient of the fit),
`own_half` (the half-width by the own error of the bin), `restored` and `restored_half` (the bin after the
correction), `law` (the value of the law), `in_law`, `status` (`law`, `reference`, `outlier` for a bin clipped
from the law, or `excluded` for a bin without a coefficient), `requests`, `effective`, `trials` and `failures`.

`fit.spread` has `parameters` (how many parameters the spread adds), `loglik_gain` ($2\Delta\ln L$ against one variance) and
`V`, `P`, `M`, the groups of the three axes. A group has `bins`, `center` (the geometric mean of the load of its requests), `share`
(its share of the requests), `failing` (its requests with failures), `sigma2` (the variance with the other two axes averaged over
the requests) and `half` (the half-width of the variance).

`spread_laws` has `V`, `P`, `M`: for every axis `form` (`line` for $a + b\log_2 x$, `hyperbola` for $c + A/(1 + x/x_0)$, or
`inverse` for $c + A/x$, the degenerate hyperbola), `params` (`a`, `b`; `c`, `A`, `x0`; or `c`, `A`), `chi2` (by the residuals
in $\ln\sigma^2$) and `n_points` (the groups). An axis with no more groups than the parameters of its form is absent.

A capacity row has `key`, `level` (0.5 for every capacity), `method` (`law` or `bins`), `value`, `status`, and
for a capacity by the law `interval` and `plus_minus`, for a capacity by the bins beyond the measured area
`lower_bound` and `reserve_left`. The meaning of every quantity: [passport model](05-passport-model.md).

## Your own measurement

To compute a passport of your own points, put them into a directory with the same two data
files. `points.jsonl.gz` may hold the records in any order, one per point, and may be split into
numbered parts as above. `measurement.json`
needs `model`, `protocol_version`, `vocab_id` and `vocab_hash`; `tariff` may be `null`, and `format_version` may be left out in the description and in the records: it reads as 1. List the
SHA-256 of `points.jsonl.gz`, or of every part, under `files` for `verify` to pass; `build_passport.py` does not
need it.
