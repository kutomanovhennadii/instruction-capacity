# IEC — instruction capacity of language models

IEC studies the behavior and degradation of a language model under controlled load. The load
is set by the simplest independent instructions a program can check: *replace word A with
word B*. A task gives the model a list of such rules and a text without meaning, made of words
from a fixed vocabulary; the model rewrites the text. Each kind of load is set separately: the
text length, the word position in the answer, the replacement density (the share of words in
the text that must be replaced), the number of rules and the position of a rule in the list.
The answer is aligned with the reference by places, and three quantities are counted on them:
`P_sync`, the answer word stands against its reference word; `H`, a rule word on its place is
replaced correctly; `F`, a background word on its place gets the replacement of a rule. Every
place is a Bernoulli trial. From many trials a statistical model recovers the laws by which each
quantity depends on each kind of load. The laws and the parameters derived from them with
uncertainties — the instruction capacity (the IEC), the fatigue capacity and the measure of the
density, how many times the full density multiplies the odds of an error — form the **passport**
of the model under measurement, together with the way the spread of difficulty between requests
depends on the load.

This repository is the public part of the instrument, version 0.2.2. With it you can

- recompute the passports of the published measurements from their points and check that the
  published files are intact and reproducible;
- rebuild the vocabulary of the published measurements and check it against its reference;
- measure points on your own model and compute its passport.

The points of the published measurement were chosen by an adaptive planning technology
developed by the author: it decides **which points to measure** from the results of the points
already measured. This technology is the closed part of the instrument and is not disclosed:
neither its design nor its rules for choosing points are published. A passport does not depend
on it: it is computed from the published points alone, whoever chose them, so every number of a
published passport can be checked without the closed part.

## Version 0.2.2

This is the second version of the instrument, release 0.2.2. The first, 0.1.0, is the version the
article describes; the tag `v0.1.0` of this repository holds it and reproduces the numbers of the
article. Release 0.2.2 publishes the measurement of GPT-4.1 mini and the passport of Nemotron 3 Ultra
550B A55B without its points, keeps the fits of a passport in a file so that a change of the laws
recomputes it in seconds, and checks a published measurement so that a fit at the edge of its
convergence does not decide the check. Release 0.2.1 corrected how the passport of 0.2.0 chooses and fits its laws: every law takes
the form with the smallest `QAIC` among its candidates, the slices of the grids choose their form too,
a point of a law weighs `1/(σᵢ² + τ²)`, the amplitudes and slopes of the curves take either sign, a
quantity whose variance by the load turns a bin against the median request of the data is fitted with
one variance, and a capacity beyond the largest floating-point number is written as such. The points of
the published measurements are those of 0.2.0, and their passports are recomputed.

Version 0.2 keeps the task: the prompt, the text, the vocabulary `v1_intersection` and the format
of the point records are those of 0.1.0, and the protocol version is `0.2`. It changes how an answer
is read and what the passport describes:

- **Reading of an answer.** Version 0.1.0 matched an answer with the reference line by line; version
  0.2 aligns it by places, so skipped pieces of text, returns, repeats, glued and split lines are seen
  and counted in words ([answer validation](docs/04-answer-validation.md)).
- **Quantities.** Version 0.1.0 counted one outcome, a right replacement on a rule word; version 0.2
  counts three, `P_sync`, `H` and `F`, which combine as `P_correct = P_sync · [π · H + (1 − π) · (1 − F)]`.
- **Models and laws.** Every quantity has its own model, laws, level and capacities
  ([passport model](docs/05-passport-model.md), [empirical laws](docs/07-empirical-laws.md)). The
  loss of the place in the text is `P_sync`, so the laws of the collapse of the answer are gone.
- **Spread of request difficulty.** Version 0.1.0 had one variance of the shift of a request. In
  version 0.2 the model is a GLMM with heterogeneous random-intercept variance: the variance depends on
  the load of the request, its text length, density and number of rules, and its groups are a part
  of the passport.
- **Capacities.** The measure of the density, `e^(−P̂(1))`, replaces the guaranteed capacities. All
  three quantities have the instruction and the fatigue capacities and the measure of the density.
  All capacities are taken at the level 0.5, where they do not depend on the spread of request
  difficulty.
- **One standard of the passport.** Every law goes through every bin with a coefficient, each bin
  weighing `1/(σᵢ² + τ²)`; a bin leaves a law only as an outlier beyond three standard deviations of
  the residuals (the procedure of astropy). Every law takes the form with the smallest
  `QAIC = χ²/ĉ + 2k` among the candidates of its block, and the established form of the block wins a
  tie ([passport model](docs/05-passport-model.md)). A call that does not fit the time limit is retried
  once with twice the limit, an answer without a word once; a second failure is `api_error` with the
  code of its reason.
- **Measurements, vocabularies, commands.** The published measurement of GPT-4o mini is new and
  contains the points of the measurement of 0.1.0; the measurements of Gemini 3.5 Flash Lite,
  Gemini 3.1 Flash Lite, Nemotron 3.5 Lightning, Nemotron 3 Nano Omni 30B A3B Reasoning and GPT-4.1
  mini are published with it. Three more vocabularies are published with their build configs and
  tokenizers. `build_passport.py` and `verify.py` work with the new `passport_data.json`.

The numbers of a passport of version 0.2 are not those of the article: the answer is read differently and
the quantities are others. The full list of changes is in the [CHANGELOG](CHANGELOG.md).

## Article

H. Kutomanov. *Instruction Capacity: Behavior and Degradation of Language Models under
Controlled Load.* arXiv:2610.36095, 2026. [arXiv page](https://arxiv.org/abs/2610.36095) ·
[PDF](articles/instruction-capacity.pdf)

The article describes version 0.1.0 of the instrument; the tag `v0.1.0` of this repository
reproduces its numbers, and [Version 0.2.2](#version-022) above says what this version changes.

## Quick start: recompute a published passport

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # the exact library versions the passports were checked with
pip install --no-deps .

python scripts/build_passport.py artefacts/pilot_gpt4o_mini_v55 --out-dir my_passport
python scripts/verify.py artefacts/pilot_gpt4o_mini_v55
```

`requirements.txt` pins every library to the version in which the published passports were
recomputed; `pyproject.toml` gives the ranges the code is written against. With other versions
the recomputation may differ in the last digits, which is what the tolerance of `verify` is for.

`poetry.lock` holds the same versions for Poetry: `poetry install --with test` builds the same
environment and installs the package.

`build_passport.py` reads the published points (`points.jsonl.gz`, or its numbered parts
`points-001.jsonl.gz`, `points-002.jsonl.gz` and so on when one file would exceed the 100 MiB limit of a
file on GitHub) and the description of the
measurement (`measurement.json`), aligns every answer, writes `passport_data.json`, `passport.md`
and the figures of the three quantities into `my_passport/` and prints the capacities it computed.
The numbers equal the published `artefacts/pilot_gpt4o_mini_v55/passport_data.json` up to the rounding of
floating-point arithmetic. `verify.py` runs four checks and prints `RESULT: PASSED`: the SHA-256 of
the files listed in `measurement.json`; the vocabulary against its reference; the rule list and the
source text of every point rebuilt from its seeds; the passport recomputed from the points against
the published one, within the [tolerance of computation](docs/05-passport-model.md#9-verification).
The model is not called; a recomputation takes a few minutes and runs a process per processor,
which `--workers` changes; the result is the same with any number of processes. `--save-fits fits.json`
also keeps the fits of the three quantities in a file, and `--from-fits fits.json` recomputes the laws,
the capacities and the document from it in seconds, for instance after a change of
`config/passport_laws.yaml` ([data format](docs/06-data-format.md#saved-fit)).

## Published measurements

| Measurement | Model | Points | Trials of `P_sync` · `H` · `F` | Instruction capacity of `H` at q = 0.5, rules: by law · binned |
|---|---|---|---|---|
| [`pilot_gpt4o_mini_v55`](artefacts/pilot_gpt4o_mini_v55/passport.md) | `gpt-4o-mini-2024-07-18` | 5,172 | 22,598,814 · 1,155,179 · 10,213,582 | 36.4 (32.8–40.4) · 35.3 |
| [`pilot_gemini3_5_flash_lite_v58`](artefacts/pilot_gemini3_5_flash_lite_v58/passport.md) | `gemini-3.5-flash-lite` | 3,788 | 48,842,597 · 7,817,448 · 24,081,763 | 14.6 (13.4–15.9) · 13.1 |
| [`pilot_gemini3_1_flash_lite_v57`](artefacts/pilot_gemini3_1_flash_lite_v57/passport.md) | `gemini-3.1-flash-lite` | 5,336 | 112,208,407 · 4,318,887 · 25,706,292 | 18.8 (15.3–23.2) · 14.8 |
| [`pilot_nemotron3_5_lightning_v43`](artefacts/pilot_nemotron3_5_lightning_v43/passport.md) | `nvidia/nemotron-3.5-lightning:free` | 4,753 | 48,574,151 · 590,039 · 8,872,340 | 10.0 (8.0–12.6) · 7.2 |
| [`pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45`](artefacts/pilot_nemotron3_nano_omni_30b_a3b_reasoning_v45/passport.md) | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 3,695 | 43,142,085 · 829,043 · 7,689,136 | 34.3 (29.8–39.4) · 35.0 |
| [`pilot_gpt4_1_mini_v56`](artefacts/pilot_gpt4_1_mini_v56/passport.md) | `gpt-4.1-mini-2025-04-14` | 6,961 | 85,962,046 · 3,805,051 · 36,982,696 | 23.5 (20.9–26.4) · 20.5 |

Intervals are 95%. The measurement of `gpt-4o-mini-2024-07-18` is the evidence base of the
[empirical laws](docs/07-empirical-laws.md). The directory of every measurement holds the points, the description,
the passport (`passport_data.json`, `passport.md`, `figures/`) and `passport.pdf` — the same passport as a PDF.

The passport of `nvidia/nemotron-3-ultra-550b-a55b:free`, `pilot_nemotron3_ultra_550b_a55b_v60` (6,702 answered
points; instruction capacity of `H` at q = 0.5: 17.0 (15.9–18.1) by law, 16.4 binned), is published without its points
in [`passports/`](passports/pilot_nemotron3_ultra_550b_a55b_v60/passport.md). Its answers were generated through the free
endpoint that NVIDIA serves on OpenRouter, and the NVIDIA API Trial Terms of Service of that endpoint do not allow the
generated content to be distributed, so this passport cannot be recomputed from published points.

## Measure your own model

A point is one call of the model. Six numbers fix its task together with the vocabulary: `D`
(distinct words of the text), `n` (occurrences of each word), `m` (rules) and the seeds
`seed_rules`, `seed_text`, `seed_order` ([task construction](docs/03-task-construction.md)).

```bash
export OPENAI_API_KEY=...          # or put the key into a .env file
python scripts/measure_point.py --model gpt-4o-mini-2024-07-18 --tokenizer gpt-4o-mini \
    --max-tokens 16384 --context-window 128000 --response-ceiling 16384 \
    --D 16 --n 5 --m 8 --seed-rules 1 --seed-text 2 --out my_points.jsonl
```

The command builds the task, checks the limits of the model before the call, calls the model,
appends the record of the point to `my_points.jsonl` and prints the counts of the alignment of its
answer. A dry run of the whole chain without a network or a key: `--provider mock --model echo
--max-tokens 4096` (the mock model echoes the prompt back, so its record is not a measurement).

To compute the passport of your model:

1. choose the points: a model fits a free coefficient per cell of its grid and per bin of its axes,
   and the bins and weights come from the points you give ([passport model](docs/05-passport-model.md));
2. measure them, one record per point;
3. make a measurement directory with the records compressed by gzip as `points.jsonl.gz` and a
   `measurement.json` ([data format](docs/06-data-format.md));
4. run `python scripts/build_passport.py <directory>`.

The bundled adapter speaks the OpenAI API. Another provider is one adapter class and one entry
in the provider registry ([`src/iec/model_client/`](src/iec/model_client/README.md)).

## The vocabulary

`assets/vocab/v1_intersection.json` holds 2220 English words, each a single token for the
tokenizers of gpt-4o, gpt-4o-mini, gpt-3.5-turbo and Qwen2.5 ([vocabulary](docs/02-vocabulary.md)).
Three more vocabularies are built by the same pipeline for other families of models:
`v1_nemotron`, `v1_gemma4` and `v1_gemini35fl`.

```bash
python scripts/verify_vocab.py                         # the file matches its reference
python scripts/build_vocab.py --assets-dir my_vocab    # rebuild it from the corpora
python scripts/build_vocab.py --config config/vocab/v1_nemotron.yaml --assets-dir my_vocab
```

A rebuild prints the hash of the vocabulary: it must equal the hash in
`assets/vocab/<vocab_id>.reference.yaml`. The rebuild downloads the NLTK corpora and the tokenizer
files on first use. A vocabulary read from the count of a Gemini model needs `GEMINI_API_KEY`.

## Documentation

1. [Instruction capacity: the unit and the experiment](docs/01-instruction-capacity.md)
2. [The vocabulary](docs/02-vocabulary.md)
3. [Task construction](docs/03-task-construction.md)
4. [Answer validation](docs/04-answer-validation.md)
5. [The passport model](docs/05-passport-model.md)
6. [Data format](docs/06-data-format.md)
7. [Empirical laws](docs/07-empirical-laws.md)

## Repository layout

```text
src/iec/        the package; every folder has a README
requirements.txt, requirements-test.txt, poetry.lock   exact library versions (runtime, test runner)
scripts/        measure_point.py, build_passport.py, verify.py, build_vocab.py, verify_vocab.py
tests/          tests of the package, mirroring src/iec/; tests/speed/ times the long operations
config/         vocab.yaml and vocab/ - how the vocabularies are built; passport_laws.yaml and
                passport_figures.yaml - the forms of the laws and the figures of the passport;
                observability.yaml - the telemetry of the bench
assets/vocab/   the vocabularies and their references
artefacts/      the published measurements
passports/      a passport published without its points
articles/       the article as a PDF
docs/           the method
.github/        the checks of a reader on every push
CHANGELOG.md    what changed between versions
```

Tests: `pip install -r requirements-test.txt`, then `python -m pytest`. Tests that call a live
provider or download tokenizer files are marked `network`, and the timing tests of the long
operations are marked `speed`; both are excluded by default.

## License

- Code — `src/`, `scripts/`, `tests/`, `config/`, `pyproject.toml`: MIT License, see
  [LICENSE](LICENSE).
- Data and documents — `artefacts/`, `passports/`, `articles/`, `assets/vocab/`, `docs/`, this README: Creative
  Commons Attribution 4.0 International, see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).
- The vocabulary is derived from the CMU Pronouncing Dictionary and WordNet; their notices are
  in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The corpora are not redistributed: the
  build code downloads them through NLTK.
- The model answers in `artefacts/` are published here as part of the data. Those of
  `gpt-4o-mini-2024-07-18` and `gpt-4.1-mini-2025-04-14` were generated through the OpenAI API: under OpenAI's terms for API
  customers the customer owns the output (OpenAI Services Agreement, section 4.1). Those of
  `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` were generated through the Gemini API: under
  the Gemini API Additional Terms of Service, Google won't claim ownership over generated content
  (section "Use of Generated Content"). Those of `nvidia/nemotron-3.5-lightning` were generated
  through OpenRouter: under the OpenRouter Terms of Service the rights in an output are set by the
  terms of its model (section 6.1), and the model is released under the OpenMDW License
  Agreement, version 1.1, which "does not impose any restrictions or obligations with respect to
  any use, modification, or sharing of any outputs generated by using the Model Materials". Those
  of `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` were generated through OpenRouter as well:
  the rights in an output are set by the terms of its model, and the model is governed by the
  NVIDIA Open Model Agreement, under which "NVIDIA does not claim ownership to any outputs
  generated using the Works or Derivative Works".
