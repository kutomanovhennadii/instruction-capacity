# point_execution — one measurement point and its record

Executes one point: builds the task, checks the model limits, calls the model, validates the
answer and returns the full outcome; reads and writes published point records.

## What is here

- `point_executor.py` — `PointExecutor`: the chain task -> ceiling check -> call ->
  validation -> record. `ceiling_verdict()` decides before the call whether the point fits
  the model limits: the prompt plus the answer reserve `max_tokens` must fit the context
  window, and the reference plus `WRAPPER_RESERVE_TOKENS = 64` must fit both `max_tokens` and
  the model response ceiling.
- `record_io.py` — `append_record()` appends one record as one JSON line;
  `read_records()` reads a JSON-lines file (`.gz` is unpacked).
- `record_task.py` — `task_from_record()` rebuilds the rule table and the source text of a
  published point; `replacement_positions()` restores the replacement markup: a position is
  replaced when its word stands on the left side of a rule.

Key signatures:

```python
PointExecutor(client: ModelClient, tokenizer: Tokenizer, vocab: Vocab, config: ModelConfig)
PointExecutor.execute(conditions: PointConditions) -> PointOutcome
ceiling_verdict(config: ModelConfig, prompt_len_tokens: int, response_tokens_needed: int) -> str | None
append_record(path: Path, record: PointRecord) -> None
read_records(path: Path) -> list[PointRecord]
task_from_record(record: PointRecord, vocab_id: str, tokenizer: Tokenizer | None = None)
    -> tuple[RuleTable, SourceText]
```

## The point record

One JSON object per line:

| JSON field | Meaning |
|---|---|
| `D`, `n`, `m` | point conditions |
| `seed_rules`, `seed_text`, `seed_order` | the three seeds |
| `rules` | the rule list: an array of pairs `[word, replacement]` in the order sent to the model |
| `source_text` | the source text in lines of 10 words, as sent to the model |
| `answer` | the raw model answer; `null` if the model was not called or did not answer |
| `status` | `ok` — the model answered (an empty answer and silence until the time limit are answers too); `na` — the task does not fit the model limits, no call was made; `api_error` — the provider did not answer |
| `prompt_tokens`, `response_tokens` | provider counters; `null` if the provider did not report them |

The prompt, the reference and the validation result are not in the record: the code here
rebuilds them from the conditions and the vocabulary.

## Statuses and boundaries

A bad answer is data: an empty or garbage answer is `ok` with honest zeros in the trials.
Silence until the time limit (`PROVIDER_TIMEOUT`) is `ok` with an empty answer — the
validator reads it as a collapse from the first word. Any other provider failure is
`api_error` without trials. An exception of the caller's own inputs (inconsistent
conditions, a broken vocabulary) is not caught here. The seeds of a point are chosen by the
caller; the executor writes no files itself.

Method: [Task construction](../../../docs/03-task-construction.md), [Data format](../../../docs/06-data-format.md).
