# point_execution — one measurement point and its record

Executes one point: builds the task, checks the model limits, calls the model and returns the full outcome; appends and
reads published point records and rebuilds the task of a record.

## What is here

| Module | Role |
|---|---|
| `point_executor.py` | `PointExecutor`: the chain task -> ceiling check -> call -> record; `ceiling_verdict()` decides before the call whether the point fits the model limits |
| `record_io.py` | `append_record()` appends one record reliably as one JSON line; `read_records()` reads a file of records (`.gz` is unpacked) |
| `record_task.py` | `task_from_record()` rebuilds the rule table and the marked source text of a published point |

## Key signatures

```python
PointExecutor(client: ModelClient, tokenizer: Tokenizer, vocab: Vocab, config: ModelConfig, *,
              telemetry: TelemetryHub | None = None)
PointExecutor.execute(conditions: PointConditions) -> PointOutcome
ceiling_verdict(config: ModelConfig, prompt_len_tokens: int, response_tokens_needed: int) -> str | None
append_record(path: Path, record: PointRecord) -> None
read_records(path: Path) -> list[PointRecord]
task_from_record(record: PointRecord, vocab_id: str, tokenizer: Tokenizer | None = None)
    -> tuple[RuleTable, SourceText]
```

## The limits of the model

`ceiling_verdict()` requires that the prompt plus the answer reserve `max_tokens` fit the context window and that the
reference plus `WRAPPER_RESERVE_TOKENS = 64` fit both `max_tokens` and the response ceiling of the model. A point that does
not fit is `na`, and the model is not called.

## The point record

One JSON object per line:

| JSON field | Meaning |
|---|---|
| `format_version` | the version of the record format; a record without it is of version 1 |
| `D`, `n`, `m` | point conditions |
| `seed_rules`, `seed_text`, `seed_order` | the three seeds |
| `rules` | the rule list: an array of pairs `[word, replacement]` in the order sent to the model |
| `source_text` | the source text in lines of 10 words, as sent to the model |
| `answer` | the raw model answer; `null` if the model was not called or did not answer |
| `status` | `ok` — the model answered with at least one word; `na` — the task does not fit the model limits, no call was made; `api_error` — the provider did not answer, the model stayed silent until the doubled time limit, or it answered twice without a word |
| `prompt_tokens`, `response_tokens` | provider counters; `null` if the provider did not report them |

The prompt, the reference and the alignment are not in the record: the code rebuilds them from the conditions and the
vocabulary. `task_from_record()` restores the replacement markup by the rule of the text generator,
`replacement_positions()`: a position is replaced when its word stands on the left side of a rule.

## Files of records

A file of records follows the JSON Lines format: only the line end `\n` separates the records, so the other characters that
Python counts as line ends, such as `\u2028`, may stand inside an answer. `append_record()` appends through `append_line()`
of `infrastructure/files/`: the whole line after a complete line, synced to the disk. A record cut short by a broken-off
process damages no other record: the next append starts on a new line, and `read_records()` stops on the cut record with
`ContractViolationError(POINT_RECORD_INVALID)` that names the file and the line.

## Telemetry

`PointExecutor` takes the telemetry hub of the command as the keyword argument `telemetry` and hands it to the generators of
the task. Every execution is an event of the block of observation `point_execution` with the point key: the model id, `m`
and the three seeds. An executor built without a hub writes the log line of every event only.

## Statuses and boundaries

A bad answer is data: a garbage answer is `ok`, and the alignment reads it from the record. An answer without a word
carries nothing to score: the model is asked once more, and a second one gives `api_error` with the code `EMPTY_ANSWER`
(decision 1 of the passport standard; `iec.contracts.has_no_words` is the one test of it, and the readers of the
measurements bought before skip such records by `is_empty_answer`). Silence until the time limit is retried once by the
reliable layer with the limit doubled; a second silence is `api_error` with `PROVIDER_TIMEOUT`. Any other provider failure is
`api_error` without an answer. The code of the failure leaves in `PointOutcome.error_code`. An exception of the caller's own inputs (inconsistent
conditions, a broken vocabulary) is not caught here. The seeds of a point are chosen by the caller; the executor writes no
files itself.

Method: [Task construction](../../../docs/03-task-construction.md), [Data format](../../../docs/06-data-format.md).
