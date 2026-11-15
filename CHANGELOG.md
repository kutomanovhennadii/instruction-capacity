# Changelog

## 0.2.0

The instrument now reads an answer by the alignment of places and describes behavior and degradation by three
quantities instead of one outcome.

**Answer validation.** A validator that matched an answer line by line is replaced by an alignment by places. Skipped pieces
of text, returns, repeats, glued and split lines are now seen and counted in words. Words that matched by chance after the
model lost its text are no longer counted as successes ([answer validation](docs/04-answer-validation.md)). The alignment
works on numeric arrays and whole matrices, so a badly damaged answer takes seconds instead of minutes, with the same result.

**Three quantities.** One outcome, a right replacement on a rule word, is replaced by `P_sync`, the answer word stands against
its reference word; `H`, a rule word on its place is replaced correctly; `F`, a background word on its place gets the replacement
of a rule. The formula `P_correct = P_sync · [π · H + (1 − π) · (1 − F)]` combines them. The passport now shows the behavior and
degradation not only of the words a rule covers but also of the background words.

**Models and laws.** Every quantity has its own model, laws, level and capacities ([passport model](docs/05-passport-model.md)):
`P_sync` — model VS-PM; `H` — model VS-PMR with an exponent on the slices of the grid; `F` — model V-(s/v)-PM. All answered points
enter the fit, so the area of analysis and the laws of the collapse of the answer are gone: the loss of the place is `P_sync`. The new
laws are in [empirical laws](docs/07-empirical-laws.md). The law of the density of `H` is a two-slope hyperbola, a hyperbola on two
slanted asymptotes of any slope: the profile of the density can fall from a top to both sides or rise at one end, and the slanted
hyperbola with a horizontal asymptote stays among its alternatives. Its shift is the law at `p = 2⁻¹⁰`.

**The spread by the load.** The fit is now a GLMM with heterogeneous random-intercept variance: the variance of the shift of a request depends on the load, `ln σ_j = z_j·γ`, with the free bins of the
length, the density and the number of rules; the fit finds it together with the coefficients and falls back on one variance if it does
not converge. The passport has a new part: the variance by the groups
of the three axes, with its table and figures.

**Capacities.** The measure of the density, `e^(−P̂(1))` - how many times the full density multiplies the odds of an error,
below one when it lowers them - replaces the guaranteed capacities; for `F` it is a conditional quantity, because a text has no background words at full density.
All three quantities have the instruction and the fatigue capacities and the measure of the density. The fatigue capacity of
`P_sync` is the root of the full law of its grid at a density going to zero. All capacities are taken at the level 0.5: there the threshold of the logit of the median request is zero whatever the spread, so the capacities do not depend on it.

**The passport standard.** One fit, one rule of the laws and one set of capacities for every model. The Newton step of the GLMM
takes the total derivative of the likelihood, with the moving nodes of the quadrature, and the moduli of the eigenvalues; a group
of the spread pushed to its lower bound is merged with its neighbour. Every law goes through every bin with a coefficient, and a
bin leaves a law only as an outlier: farther than three standard deviations of the residuals from their median, by the
procedure of astropy. Every block has its established form; an alternative - the line, the hyperbola, the slanted hyperbola or
the exponential - replaces it only when it is better by more than 10 of `QAIC = χ²/ĉ + 2k`. A call that does not fit the time
limit is retried once with 600 s, and an answer without a word is asked once more; a second failure is `api_error` with the code
of its reason, and the passport skips the empty answers of the measurements bought before.
A spelling term in the model of `F` was tested and is not in the passport: it barely moves the fit and takes
out of the law of the number of rules what adding a rule really does to a text.

**Measurement.** The measurement `pilot_gpt4o_mini_v55` with 5,172 answered points replaces `pilot_gpt4o_mini_v34`: its points are a part of the
new measurement, and the new points were measured for the new quantities. Four more measurements are published: `pilot_gemini3_5_flash_lite_v58`
of `gemini-3.5-flash-lite` with the vocabulary `v1_gemini35fl` (3,788 answered points), `pilot_gemini3_1_flash_lite_v57` of
`gemini-3.1-flash-lite` with the vocabulary `v1_gemma4` (5,336 answered points), `pilot_nemotron3_5_lightning_v43` of
`nvidia/nemotron-3.5-lightning:free` with the vocabulary `v1_nemotron` (4,753 answered points) and
`pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45` of `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` with the vocabulary
`v1_nemotron` (3,695 answered points).

**Commands.** `build_passport.py` and `verify.py` work with the new passport: `passport_data.json` has a new composition, and the tolerance
of `verify` is measured anew. `measure_point.py` prints the counts of the alignment. The protocol version is `0.2`: the prompt and the text
are those of `0.1`, and the point records have the same format. `build_passport.py` and `verify.py` take the forms of the laws from
`config/passport_laws.yaml` (`--laws`), `build_passport.py` takes the plan of the figures from `config/passport_figures.yaml`
(`--figures`), and both compute in a pool of processes (`--workers`, one per processor by default) with the same result as in one
process. `passport_data.json` carries its `format_version`. Both commands read the points of a measurement from `points.jsonl.gz`
or from its numbered parts `points-001.jsonl.gz`, `points-002.jsonl.gz` and so on, in the order of their numbers: points that would
make one file larger than the 100 MiB limit of a file on GitHub are kept in parts, as those of `pilot_gemini3_5_flash_lite_v58` and
`pilot_gemini3_1_flash_lite_v57`.

**Code.** The package is a tree of folders by responsibility, every folder with a README. The forms of the laws of the passport
are a collection of their own, and `config/passport_laws.yaml` assigns a form to every block of every quantity; the figures are
drawn by kinds of figures, and `config/passport_figures.yaml` says which figures every quantity gets. Files are written whole or
not at all, through a temporary file and a rename. The telemetry of the bench goes to a file, the console or OpenTelemetry
(`pip install "iec[opentelemetry]"`). `poetry.lock` holds the versions of `requirements.txt` and `requirements-test.txt` for
Poetry, and the checks of a reader run on every push.

**Vocabularies.** Three vocabularies for other families of models are published next to `v1_intersection`: `v1_nemotron`, `v1_gemma4` and
`v1_gemini35fl`, with their build configs in `config/vocab/`. The tokenizers `hfjson` and `gemini_count` are in the package, and `build_vocab.py`
writes the reference of a version, so anyone can rebuild a vocabulary and compare its hash with the reference ([vocabulary](docs/02-vocabulary.md)).

**The article.** The article arXiv:2610.36095 describes version 0.1.0; the tag `v0.1.0` reproduces its numbers.
