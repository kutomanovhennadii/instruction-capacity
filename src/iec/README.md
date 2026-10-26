# iec — the public measurement core

The package measures points of a language model and computes the passport of a measurement
from its points. It is everything a reader needs to check the published measurements and to
measure a model of their own.

## Contents

| Folder | Role |
|---|---|
| `contracts/` | the data of the core: conditions of a point, rules, text, answer, trial, point record, measurement description |
| `infrastructure/` | exceptions with error codes, logging, telemetry |
| `vocab/` | the vocabulary: building, loading by hash, checking against its reference |
| `prompt_generator/` | the task: rule list, text and prompt from the conditions and the seeds |
| `model_client/` | models and tokenizers behind one contract; the reliable call layer |
| `answer_validator/` | the reference answer and the Bernoulli trials of an answer |
| `point_execution/` | one point: task, limits of the model, call, validation, record |
| `passport/` | the passport of a measurement and `verify` |

## Key entry points

```python
iec.__version__                                   # "0.1.0"
iec.point_execution.PointExecutor(client, tokenizer, vocab, model_config).execute(conditions)
iec.passport.passport_data.build_passport_data(description, records, name)
iec.passport.verify.verify_measurement(directory, vocab_dir)
```

The path of a point: conditions → task (`prompt_generator`) → limits and call (`model_client`)
→ validation (`answer_validator`) → record (`contracts.PointRecord`). The path of a passport:
records → trials → area of analysis and bins → Bernoulli fit → second stage → documents
(`passport`).

## Boundaries

The package does not choose which points to measure: a measurement driver gives it the
conditions of every point. The command-line entry points are in `scripts/`.

Method: [documentation](../../README.md#documentation).
