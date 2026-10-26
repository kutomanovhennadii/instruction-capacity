# Data format

A published measurement is a directory `artefacts/<measurement>/`:

| File | Content |
|---|---|
| `measurement.json` | the description of the measurement |
| `points.jsonl.gz` | the records of the points, gzip-compressed JSON lines |
| `passport_data.json` | the passport: every number with its standard error, the full coefficients and covariance |
| `passport.md`, `figures/` | the passport as a document, with ten figures of the model in `figures/vspmr_glmm_v1/`, four figures of the rate of the collapse of the answer `figures/collapse-*.png` and six of its place `figures/place-*.png` |
| `passport.pdf` | the same document as a PDF |

The data are what was sent to the model and what it answered, plus the numbers that rebuild the
task. What the code computes from them (the prompt, the reference answer, the trials) is not
stored.

## Point record

One JSON object per line:

| Field | Type | Meaning |
|---|---|---|
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
| `model` | `model_id` (as named by the provider), `provider`, `gen_params` (`temperature`, `max_tokens` = the answer reserve, `top_p`, `seed`), `tokenizer_name`, `context_window_tokens`, `response_ceiling_tokens` (`null`: not declared and not checked) |
| `protocol_version` | the version of the protocol: prompt template, text layout, validation rules; `"0.1"` |
| `vocab_id`, `vocab_hash` | the vocabulary and its hash |
| `tariff` | `input_usd_per_million`, `output_usd_per_million`: the price the points were paid at; `null` if the money is not computed |
| `files` | file name → SHA-256 of the data file `points.jsonl.gz` |

The passport and the PDF are not listed: `verify` recomputes the passport, and a PDF is not
comparable byte for byte.

## Passport data

The keys of `passport_data.json`: `model` (the model id and the name of the measurement); `fit`
(the coefficients θ with their standard errors and 95% intervals, the full covariance,
$\hat\sigma^2$ with its standard error `sigma2_se` and its origin `sigma2_method` (`inside the
fit`), `shift_in_fit` (the shift of a point is inside the likelihood), $\hat\rho$, the sizes and
weights of the fit); `bernoulli_coefficients` (the table of coefficients); `passport` (the bins of
every block with their statuses and the second stage: laws, shifts, $\alpha_{\mathrm{phys}}$,
restored bins, public quantities, plateau by length slice, component correlations); `collapse`
(the collapse of the answer over all answered points: how many lost the text, the weights, the
bins of the text length and of the density with their covariance, the laws of the two axes, their
shifts, the restored bins, the rate law with its covariance, the check bins of the number of
rules with their Wald statistic; and `place` - the same for the place of the collapse over the
words of the points that lost the text, with the axes word number, density and share of the
text, the place law and the checks of the text length and of the number of rules; `null` when
the outcome does not vary); `resources`
(points, answered points, trials, provider tokens, money at the tariff); `bin_cube` (the filling
of the purchase cube as text); `alarms`. The blocks `fit`, `bernoulli_coefficients` and
`passport` are keyed by the passport model; a published passport has one, `vspmr_glmm_v1`. The
meaning of every quantity: [passport model](05-passport-model.md).

## Your own measurement

To compute a passport of your own points, put them into a directory with the same two data
files. `points.jsonl.gz` may hold the records in any order, one per point. `measurement.json`
needs `model`, `protocol_version`, `vocab_id` and `vocab_hash`; `tariff` may be `null`. List the
SHA-256 of `points.jsonl.gz` under `files` for `verify` to pass; `build_passport.py` does not
need it.
