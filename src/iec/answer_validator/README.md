# answer_validator — the reference answer and the alignment of an answer

Builds the reference answer `Y* = apply(R_m, X)` deterministically and aligns the raw model answer with it by places. The places
give the trials of all three quantities of the passport, so the alignment is an axiom of the technology: a change of the code
keeps its result byte for byte.

## What is here

| Folder or module | Role |
|---|---|
| `reference_builder.py` | `build_reference` and `rebuild_reference`: the reference answer of a task |
| `alignment/` | `align_answer`: the alignment of an answer with the reference by places, in steps |

## Key signatures

```python
build_reference(rule_table: RuleTable, source_text: SourceText) -> Reference
rebuild_reference(rule_table: RuleTable, source_text: SourceText) -> Reference
align_answer(answer: str, reference: Reference, source_text: SourceText, rule_table: RuleTable) -> AlignedAnswer
```

The package root exports what a consumer needs: `align_answer`, `AlignedAnswer`, `Place`, the kinds of places `PLACE_MATCHED`,
`PLACE_DELETED` and `PLACE_INSERTED` with their numeric codes `KIND_MATCHED`, `KIND_DELETED` and `KIND_INSERTED`, the origin
`GROUP_ANCHOR`, and the text forms of an answer `normalize_output` and `split_lines`.

## The reference

`build_reference` is a pure function: the mapping `{source -> target}` is applied to the stream element by element, background
passes as is. A built-in check against the replacement markup raises `ContractViolationError(REFERENCE_MARKUP_MISMATCH)` on
inconsistent inputs: markup outside the stream, a marked word not from the rules, a replaced position not marked (or the
reverse), an identity rule. That is a fault of the caller, not a property of the answer.

`rebuild_reference` gives the same reference without the check and without a log line, for a task rebuilt from its own record:
the markup of such a task follows from its words.

## Boundaries

Provider failures (timeout, API error) never reach the alignment: the point executor separates them. The trials of the three
quantities are counted from the places by the passport.

Method: [Answer validation](../../../docs/04-answer-validation.md).
