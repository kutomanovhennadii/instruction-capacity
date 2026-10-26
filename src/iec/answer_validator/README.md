# answer_validator — reference and answer validation

Builds the reference answer `Y* = apply(R_m, X)` deterministically and compares the raw model
answer `Y` with it trial by trial.

## `reference_builder` — the reference

```python
build_reference(rule_table: RuleTable, source_text: SourceText) -> Reference
```

A pure function: the mapping `{source -> target}` is applied to the stream element by
element, background passes as is. A built-in check against the replacement markup raises
`ContractViolationError(REFERENCE_MARKUP_MISMATCH)` on inconsistent inputs: markup outside
the stream, a marked word not from the rules, a replaced position not marked (or the
reverse), an identity rule. That is a fault of the caller, not a property of the answer.

## `run_validator` — the validator

```python
RunValidator().validate(
    raw_output: RawOutput, reference: Reference, source_text: SourceText, rule_table: RuleTable
) -> TrialResult
```

Order: normalization (only trailing whitespace; case matters) -> `p_strict` byte for byte
-> stripping of the edge wrapper -> line-by-line matching -> verdict `z` and category for
each marked position -> counter of damaged background.

| Category | What stands in the trial position | `z` |
|---|---|---|
| `correct` | the expected replacement | 1 |
| `unreplaced` | the original word — the rule is lost | 0 |
| `wrong_rule_target` | the replacement of **another** rule — rules are mixed up | 0 |
| `other_word` | any other word | 0 |
| `missing` | the position is unreachable: the line structure broke earlier | 0 |

**A bad answer is data, not an exception.** An empty answer, a fragment, garbage give zeros
in `z` and a valid `TrialResult`. `ValidationError(VALIDATOR_INPUT_MISMATCH)` is raised only
on inconsistent inputs of the caller.

**Edge wrapper.** Models often frame the answer with their own remarks ("Here is the
rewritten text", a code fence). Two independent stages look for the core: the protocol
alphabet (lines outside lowercase ASCII words are stripped from the edges) and the edge
anchors (the core starts with `Y*[0]` or `X[0]` and ends with `Y*[v-1]` or `X[v-1]`; only
lines of non-protocol width are stripped). A candidate is accepted only with exactly the
reference number of lines; otherwise the answer is validated whole. The violation of form is
recorded in `wrapper_leading_lines`, `wrapper_trailing_lines`, `wrapper_stripped` and always
excludes `p_strict`. The middle of the answer is never touched.

**Reachability by line length, not by word match.** A line with the reference word count is
scored position by position. In a line with another count the matched prefix is kept and
the tail is unreachable; the next line is synchronized anew. Damage costs the tail of one
line, not of the text.

**Failure ladder.** The 2nd kind — damage absorbed by its line (the next line is
recognizable by its word count). The 3rd kind — damage not absorbed, or the number of lines
differs from the reference; it is recorded on the line where the answer loses synchronization
with its text. Answer lines are tied to the reference by exact copies, forward through the
text: if a tie with a shifted number occurs, the place is the first line after the last tie
at its own number; otherwise it is the first unabsorbed damaged line or the first line beyond
the shorter side, and it is never later than them. The two kinds can coexist in one result.

## Boundaries

Provider failures (timeout, API error) never reach the validator: the point executor
separates them. Aggregates derived from the trials (`h`, `p_event`, `err_rate`, `p_rule`) are
properties of the contracts, not a separate metrics module.

Method: [Answer validation](../../../docs/04-answer-validation.md).
