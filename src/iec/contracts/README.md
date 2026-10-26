# contracts — data on the boundaries of the layers

Passive, immutable data structures exchanged by the layers. Contracts read no files and no
configs, write no logs and contain no measurement logic. All of them are
`@dataclass(frozen=True, slots=True)`; those that are stored have `as_dict()` and, where
they are read back, `from_dict()`.

## Point and measurement

- `PointConditions` — the six numbers that fully determine the task of one point: `D`
  (distinct words of the source text), `n` (occurrences of each), `m` (rules), `seed_rules`
  (draws the rule list), `seed_text` (shuffles the source text), `seed_order` (orders the
  rule lines; `0` keeps the drawn order). The text length is the property `v = D * n`.
- `PointRecord` — the published record of one point (fields in `point_execution/README.md`);
  `PointStatus` — `ok`, `na`, `api_error`. Reading is tolerant to unknown keys; a missing
  key or an unknown status raises `ContractViolationError(POINT_RECORD_INVALID)`.
  `PROTOCOL_VERSION` — version of the measurement protocol (prompt template, text layout,
  validation rules).
- `PointOutcome` — what the point executor returns to the caller: the record plus everything
  built on the way (rule table, source text, prompt, reference, prompt length, raw output,
  validation result, provider error code, duration). Only the record is published.
- `ModelConfig` — the model: `model_id`, `provider`, `gen_params`, `tokenizer_name`,
  `context_window_tokens`, `response_ceiling_tokens` (`None` — not declared).
- `MeasurementDescription` — one object per measurement, common to all its points: model,
  protocol version, vocabulary and its hash, tariff, file hashes.

## Task and answer

- `GenParams` — generation parameters: `temperature`, `max_tokens`, `top_p`, `seed`.
- `Vocab` — the vocabulary `V`: `vocab_id`, `words`, `hash`, tokenizer names; properties
  `size` and `max_pairs`; `compute_hash(words)` is the canonical hash definition.
- `Rule` / `RuleTable` — the rule list: pairs `source -> target` in the order shown to the
  model, `vocab_id`, `seed_rules`; properties `m`, `sources`, `targets`.
- `SourceText` — the word stream: `words`, `words_per_line`, `replacement_positions`
  (indices of the words to replace), `len_tokens`; the text form is the property `text`.
- `Prompt` — the prompt text and an optional payload position (always `None` here).
- `RawOutput` — the raw answer `text`, `usage` (`TokenUsage`), `finish_reason`,
  `latency_s`, `raw`.
- `TokenUsage` — provider counters `prompt_tokens`, `response_tokens` (`None` — not
  reported, not zero).
- `Tariff` — price per million input and output tokens.

## Validation result

- `Reference` — the reference answer `Y*`: `words` and `words_per_line`.
- `TrialRecord` — one binary trial: `t`, `rule_index`, `word_position`, `expected`,
  `observed` (`None` — unreachable), `z` (0/1) and the category `outcome` (`correct`,
  `unreplaced`, `wrong_rule_target`, `other_word`, `missing`).
- `RuleStat` — statistics of one rule: `attempts`, `correct`, property `p_rule`.
- `TrialResult` — the summary: `records`, `rule_stats`, the failure ladder, the counters of
  broken lines and damaged background, `p_strict`, the wrapper fields. Aggregates derived
  from the records are properties: `h`, `correct_count`, `p_event`, `err_rate`.

The failure ladder ("kind" is a scale of answer damage, unrelated to type I/II errors in
hypothesis testing):

| Kind | What happened | Field |
|---|---|---|
| 1st | a wrong replacement with an intact line structure | `first_error_t`, `first_error_rule_index` |
| 2nd | a line has the wrong word count, but the next line is recognizable | `first_fail_kind_2_line` |
| 3rd | control over lines is lost: the damage is not absorbed, or the number of lines differs | `first_fail_kind_3_line` — the line where the answer loses synchronization with its text (by exact copies of lines, forward), never later than the first unabsorbed damaged line and the first line beyond the shorter side |

"Did not happen" is `None` everywhere, not `0`.

Key signatures:

```python
PointConditions(D: int, n: int, m: int, seed_rules: int, seed_text: int, seed_order: int = 0)
PointRecord.from_dict(raw: Mapping[str, Any]) -> PointRecord
ModelConfig(model_id: str, provider: str, gen_params: GenParams, ...)
Vocab.compute_hash(words: Iterable[str]) -> str
RuleTable(rules: tuple[Rule, ...], vocab_id: str, seed_rules: int)
SourceText(words: tuple[str, ...], words_per_line: int, replacement_positions: tuple[int, ...], ...)
TrialResult(records: tuple[TrialRecord, ...], rule_stats: tuple[RuleStat, ...], ...)
```

## Boundaries

Contracts check only what makes a single object invalid by itself (a broken record, an
unknown status). Consistency between objects — `m <= D`, a vocabulary large enough — is
checked by the generators, which hold both the conditions and the vocabulary.

Method: [Data format](../../../docs/06-data-format.md).
