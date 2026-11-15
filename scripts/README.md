# scripts — the command-line entry points

The commands of a reader: measuring a point, building and verifying the vocabulary, building and verifying the passport of
a measurement. A command is a thin entry point: it reads its arguments, assembles the objects of the package and prints
the summary; the work itself lives in the package. The paths `scripts/build_passport.py` and `scripts/verify.py` are
printed in the first article and stay.

## What is here

| Command | Role |
|---|---|
| `measure_point.py` | measures one point: builds the task, calls the model, aligns the answer, appends the record and prints the counts of its trials |
| `build_vocab.py` | builds a vocabulary from its config, saves it as a new version and writes the reference of the version |
| `verify_vocab.py` | checks a vocabulary file against the reference of its version |
| `verify.py` | verifies a published measurement: the checksums of its files, the vocabulary, the tasks rebuilt from their seeds and the passport recomputed with the forms of the laws of `--laws`, in a pool of `--workers` processes (by default one per processor) |
| `build_passport.py` | builds the passport of a published measurement: `passport_data.json`, `passport.md` and the figures, with the forms of the laws of `--laws` and the plan of the figures of `--figures`, in a pool of `--workers` processes (by default one per processor) |

```bash
python scripts/measure_point.py --provider mock --model mock-model --max-tokens 4096 --D 16 --n 5 --m 8 --seed-rules 1 --seed-text 2 --out my_points.jsonl
python scripts/verify_vocab.py                                    # the published vocabulary matches its reference
python scripts/build_vocab.py --config config/vocab.yaml --assets-dir my_vocab
python scripts/build_passport.py artefacts/pilot_gpt4o_mini_v55 --out-dir my_passport --laws config/passport_laws.yaml --figures config/passport_figures.yaml
python scripts/verify.py artefacts/pilot_gpt4o_mini_v55 --vocab-dir assets/vocab --laws config/passport_laws.yaml
```

## Boundaries

The commands choose no points. Only `measure_point.py` asks a model for an answer; `build_vocab.py` may ask a provider to count
the tokens of a word.
