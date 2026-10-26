# IEC — instruction capacity of language models

IEC measures how many independent instructions a language model holds and executes at once.
The instruction is the simplest one a program can check: *replace word A with word B*. A task
gives the model a list of such rules and a text made of words from a fixed vocabulary; the
model rewrites the text. Every occurrence of a word that a rule replaces is one Bernoulli
trial: 1 if the model wrote the right replacement, 0 otherwise. A statistical model fitted to
many trials gives the **passport** of the model under measurement: how many rules it holds
above a quality level (the IEC), how fast its accuracy falls along a long answer, and what a
longer text, a denser table of replacements and a longer list of rules cost.

This repository is the public part of the instrument, version 0.1.0. With it you can

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

## Quick start: recompute a published passport

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # the exact library versions the passports were checked with
pip install --no-deps .

python scripts/build_passport.py artefacts/pilot_gpt4o_mini_v34 --out-dir my_passport
python scripts/verify.py artefacts/pilot_gpt4o_mini_v34
```

`requirements.txt` pins every library to the version in which the published passports were
recomputed; `pyproject.toml` gives the ranges the code is written against. With other versions
the recomputation may differ in the last digits, which is what the tolerance of `verify` is for.

`build_passport.py` reads the published points (`points.jsonl.gz`) and the description of the
measurement (`measurement.json`), writes `passport_data.json`, `passport.md`, ten figures of
the model and ten figures of the collapse of the answer (its rate and its place) into
`my_passport/` and prints the IEC it computed. The numbers equal the published
`artefacts/pilot_gpt4o_mini_v34/passport_data.json` up to the rounding of floating-point
arithmetic. `verify.py` runs four checks and prints `RESULT: PASSED`: the SHA-256 of the files
listed in `measurement.json`; the vocabulary against its reference; the rule list and the
source text of every point rebuilt from its seeds; the passport recomputed from the points
against the published one, within the
[tolerance of computation](docs/05-passport-model.md#9-verification). The model is not called;
a recomputation takes a few minutes.

## Published measurements

| Measurement | Model | Points | Bernoulli trials | IEC at q = 0.95, rules: binned · by law |
|---|---|---|---|---|
| [`pilot_gpt4o_mini_v34`](artefacts/pilot_gpt4o_mini_v34/passport.md) | `gpt-4o-mini-2024-07-18` | 2958 | 642430 | 36.58 ± 5.41 · 38.06 ± 6.13 |

Intervals are ±1.96 standard errors. The measurement is the evidence base of the
[empirical laws](docs/07-empirical-laws.md). Its directory holds the points, the description,
the passport (`passport_data.json`, `passport.md`, `figures/`) and `passport.pdf` — the same
passport as a PDF.

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
validates the answer and appends the record of the point to `my_points.jsonl`. A dry run of the
whole chain without a network or a key: `--provider mock --model echo --max-tokens 4096` (the
mock model echoes the prompt back, so its record is not a measurement).

To compute the passport of your model:

1. choose the points: the passport fits a free coefficient per cell of its grid, and a cell of
   the purchase cube counts only with at least three executed points at its upper corner
   ([passport model](docs/05-passport-model.md));
2. measure them, one record per point;
3. make a measurement directory with the records compressed by gzip as `points.jsonl.gz` and a
   `measurement.json` ([data format](docs/06-data-format.md));
4. run `python scripts/build_passport.py <directory>`.

The bundled adapter speaks the OpenAI API. Another provider is one adapter class and one entry
in the provider registry ([`src/iec/model_client/`](src/iec/model_client/README.md)).

## The vocabulary

`assets/vocab/v1_intersection.json` holds 2220 English words, each a single token for the
tokenizers of gpt-4o, gpt-4o-mini, gpt-3.5-turbo and Qwen2.5 ([vocabulary](docs/02-vocabulary.md)).

```bash
python scripts/verify_vocab.py                         # the file matches its reference
python scripts/build_vocab.py --assets-dir my_vocab    # rebuild it from the corpora
```

The rebuild downloads the NLTK corpora and the files of the Qwen tokenizer on first use.

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
requirements.txt, requirements-test.txt   exact library versions (runtime, test runner)
scripts/        measure_point.py, build_passport.py, verify.py, build_vocab.py, verify_vocab.py
tests/          tests of the package, mirroring src/iec/
config/         vocab.yaml - how the vocabulary is built
assets/vocab/   the vocabulary and its reference
artefacts/      the published measurements
docs/           the method
```

Tests: `pip install -r requirements-test.txt`, then `python -m pytest`. Tests that call a live
provider or download tokenizer files are marked `network` and excluded by default; one test
compares the fit with `statsmodels` and is skipped when that library is not installed.

## License

- Code — `src/`, `scripts/`, `tests/`, `config/`, `pyproject.toml`: MIT License, see
  [LICENSE](LICENSE).
- Data and documents — `artefacts/`, `assets/vocab/`, `docs/`, this README: Creative Commons
  Attribution 4.0 International, see [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).
- The vocabulary is derived from the CMU Pronouncing Dictionary and WordNet; their notices are
  in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The corpora are not redistributed: the
  build code downloads them through NLTK.
- The model answers in `artefacts/` were generated by OpenAI models through the OpenAI API.
  Under OpenAI's terms for API customers the customer owns the output (OpenAI Services
  Agreement, section 4.1); the answers are published here as part of the data.
