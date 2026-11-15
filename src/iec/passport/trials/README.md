# trials — the trials of the passport and their layout

The trials of the three quantities are counted from the places of the alignment of every point and folded into matrices in
the layout of the bins of the passport. A trial of P_sync is a place of the alignment, a trial of H a rule word on its
place, a trial of F a background word on its place; the word-number bin of a trial is the doubling of its line number.

## What is here

| Module or folder | Role |
|---|---|
| `place_trials.py` | `point_trials()`: the trials of one point by word-number bins and its counts; `PointTrials`, `TOTALS` |
| `trial_tables.py` | `TrialTables` and `build_trial_tables()`: the tables "line number -> doubling bin" and "word number -> log2" of a measurement |
| `layout/` | the bins of the axes and the doubling rule, the run cube V × P × M with the weights of the points |
| `design/` | the folded design matrices of the quantities: a strategy per quantity and per axis |

## Key signatures

```python
build_trial_tables(top: int) -> TrialTables
point_trials(key: str, D: int, n: int, m: int, aligned: AlignedAnswer, reference: Reference,
             source_text: SourceText, rule_table: RuleTable, *, tables: TrialTables) -> PointTrials
```

## Speed

The tables of the trials are a structure for speed: the caller builds them once at the setup of a measurement, for its
longest text, and hands them to the counting of every point. A point with a longer text is a
`ContractViolationError(TRIAL_TABLES_INVALID)`. The folder keeps no state of the process.

## Boundaries

The folder counts and folds trials; it fits nothing: the fits come with `fitting/`. The alignment of an answer comes from
`iec.answer_validator`.
