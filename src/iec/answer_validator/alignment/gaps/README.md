# gaps — the decision of the gaps between anchors

A gap is the run of reference lines and answer lines between two anchors. Its decision pairs gap lines so that as many words as
possible match: one to one, one answer line against two reference lines (a glue) or two answer lines against one (a split),
each pairing above the copy threshold `COPY_SHARE = 0.5` of the shorter side. A copy of a reference line and a line without a
pair are insertions; a reference line without a pair is a deletion.

## What is here

| Module | Role |
|---|---|
| `gap_resolver.py` | the interface `GapResolver`, the choice by size `SizedGapResolver` and the strategy of the alignment `DEFAULT_GAP_RESOLVER` |
| `gap_decision.py` | `Group`, the copy threshold, the value of a pairing and the walk back over the table |
| `small_gap.py` | `SmallGapResolver`: the table line by line; `lcs_length` |
| `large_gap.py` | `LargeGapResolver`: the table by whole matrices |
| `word_matrices.py` | lines as numeric matrices: common words, matches by positions, the subsequence of many pairs bit by bit |

## Key signatures

```python
GapResolver.resolve(gap_reference, gap_answer, reference_signatures, answer_signatures, copies) -> GapDecision
SizedGapResolver(small: GapResolver, large: GapResolver)
DEFAULT_GAP_RESOLVER = SizedGapResolver(SmallGapResolver(), LargeGapResolver())
```

## The strategies

Both strategies fill the table `best[a][b]` - the best value of the first `a` reference lines against the first `b` answer
lines, where a pairing is worth `2·matched` half words and three less for a glue and a split - and give the same decision. The
small one counts it line by line. The large one drops most pairs of lines at once by the number of common words regardless of
order, counts the longest common subsequence of the rest bit by bit and fills every row at once. A gap of at most 16 pairs of
lines, or a gap with a reference line longer than 32 words, goes the small way: a glue of two lines must fit the 64 bits of a
machine word. A new strategy is a new class with the method `resolve`; the alignment does not change. The counts of the common
words and of the subsequences are compiled loops of `iec.infrastructure.acceleration`; without numba they run by numpy steps
with the same result.
