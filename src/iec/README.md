# iec — the public measurement core

The package measures points of a language model and computes the passport of a measurement
from its points. It is everything a reader needs to check the published measurements and to
measure a model of their own.

## Contents

| Folder | Role |
|---|---|
| `contracts/` | the data of the core: conditions of a point, rules, text, answer, point record, measurement description |
| `infrastructure/` | exceptions with error codes, logging, telemetry with its sinks, loading of YAML files, observability assembled from one configuration, reliable writing of files |
| `vocab/` | the vocabulary: building, loading by hash, checking against its reference |
| `prompt_generator/` | the task: rule list, text and prompt from the conditions and the seeds |
| `model_client/` | models and tokenizers behind one contract; the reliable call layer |
| `answer_validator/` | the reference answer and the alignment of an answer by places |
| `point_execution/` | one point: task, limits of the model, call, record |
| `passport/` | the passport of a measurement and `verify` |

## Key entry points

```python
iec.__version__                                   # "0.2.1"
iec.point_execution.PointExecutor(client, tokenizer, vocab, model_config).execute(conditions)
iec.answer_validator.align_answer(answer, reference, source_text, rule_table)
iec.passport.passport_data.measurement_trials(description, records)
iec.passport.passport_data.build_passport_data(description, records, trials, name)
iec.passport.verify.verify_measurement(directory, vocab_dir)
```

The path of a point: conditions → task (`prompt_generator`) → limits and call (`model_client`)
→ record (`contracts.PointRecord`). The path of a passport: records → alignment and trials of the
three quantities (`answer_validator`, `passport`) → bins and fits → laws, levels and capacities →
documents (`passport`).

## Boundaries

The package does not choose which points to measure: a measurement driver gives it the
conditions of every point. The command-line entry points are in `scripts/`.

Method: [documentation](../../README.md#documentation).
