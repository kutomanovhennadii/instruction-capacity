# Answer validation

The answer of the model is aligned with the reference answer by places, and the outcomes of three
quantities are counted on these places. The reference answer is computed by a program, so no
judge, human or model, and no text-similarity metric enters the measurement.

A bad answer is data, not an error of the instrument: a model that lost half of the replacements
or returned noise gives a valid measurement. Only an inconsistency of the instrument's own inputs
stops validation, and technical failures of the provider never reach it: they are the status
`api_error`.

## The reference answer

Each word of the text that stands on the left side of a rule becomes the right side of that rule;
every other word stays. One pass is enough: the words to replace and the replacements are disjoint,
and the replacements never occur in the text ([task construction](03-task-construction.md)). The
positions that change are checked against the positions marked when the text was built.

## Why an alignment

Models often lose the order of the lines. They skip a piece of the text, go back and rewrite what
they have passed, repeat one line, glue two lines together or split one in two. A comparison of an
answer line with the reference line of the same number reads such an answer against foreign lines:
it takes a word that matches by chance for a success and a word of a repeated line for a changed
background word. So the answer and the reference are aligned the way two DNA chains or a recognized
speech and its text are aligned. Deletions, insertions and repeats are then seen and counted in
words, in the same way as wrong replacements.

## Places

A **place** is the unit of the alignment: a word of the reference or a word of the answer that has
no place in the reference. There are three kinds of places.

| Kind | What it is |
|---|---|
| matched | an answer word stands against a reference word, right or not |
| deleted | a reference word has no answer word: the model skipped a piece of text or cut the answer |
| inserted | an answer word has no place in the reference: a repeat, a return, a loop or an extra word |

Every word of the reference and every word of the answer core is in exactly one place; the
instrument checks this conservation and stops on a violation. A wrong word on its place is not a
failure of the alignment but the outcome of the trial of a rule word or of a background word.

Case matters, and normalization is minimal: only the trailing whitespace of the answer is
removed. `Stone` instead of `stone` is a wrong word.

## The wrapper of an answer

Models often frame the answer, with "Here is the rewritten text:", a code fence or a closing
remark, although the task asks for the rewritten text only. The lines at the edges of the answer
without a single word of the task are the **wrapper**; they run in a row from the very edge and are
not counted. An empty line has no task words. A line with at least one task word is never a wrapper:
it is rewritten text with errors. The answer without the wrapper lines is the **core**.

## Two levels: lines, then words

**Anchors.** Lines are compared in the **canonical form**: the right-hand word of every rule is
replaced by its left-hand word. A right replacement and an unreplaced word then look the same, and
a wrong replacement or a misfire leaves the left-hand word of a rule on the place. A **background
word** is a word of the text that is no left-hand word of a rule; the **skeleton** of a line masks
the left-hand words of the rules. An answer line can stand on a text line equal to it; otherwise on
a text line with the same skeleton where more than half of the places agree; otherwise on a text
line of the same length that keeps every background word of the answer line on its place, with more
than half of the places agreeing. Of all such pairs the **anchors** are the longest chain that goes
forward both in the text and in the answer: the longest common subsequence of lines (Hunt, McIlroy
1976), with the equality of lines judged by these rules; of equal chains, the one where earlier
answer lines stand against the reference. A line from a far place of the text cannot carry the
anchors away, and when the model goes back, the first pass stays on its place and the repeat falls
into a gap.

**Gaps.** Between two neighbouring anchors lies a gap. A line of a gap that exactly copies a
reference line is an insertion: the anchors have already put the line it repeats on its place. The
other lines of the answer are paired with the reference lines of the gap so that as many words as
possible match: one to one, one answer line against two neighbouring reference lines (a **glue**) or
two against one (a **split**). A pairing is allowed only if more than half of the words of the
shorter side match, the **copy threshold**: a word of a foreign line matches a reference word by
chance with a probability of about $1/D$, a word of a rewritten line almost always. Lines left
without a pair are deleted or inserted whole.

**Words.** In a pair of lines of equal length the words are compared by their positions. Otherwise
they are aligned with the fewest edits.

## The three quantities

| Quantity | Trial | Success | Failure |
|---|---|---|---|
| `P_sync` | a word of the reference | the word stands on its place | the word is deleted |
| `H` | a rule word on a matched place | the right replacement | the word left as it was, the right-hand word of another rule, any other word |
| `F` | a background word on a matched place | no misfire | a misfire: the right-hand word of a rule on the place of the word |

A deleted rule word or background word is not a trial of `H` or `F`: its failure belongs to
`P_sync`. A choice of a wrong rule is a failure of `H`, not a quantity of its own. A background word
changed to a word that is no right-hand word of a rule is counted, but it is not the event of `F`.

The **word number** `s` of a trial is counted by the reference, from one. An inserted word has no
place in the reference and is no trial of any quantity: it is counted in `inserted`.

## What is counted for an answer

| Count | Meaning |
|---|---|
| `places`, `matched`, `deleted`, `inserted` | the places by their kinds |
| `wrapper_lines` | lines of the wrapper |
| `episodes` | gaps with a deletion or an insertion: failures of synchronization |
| `rule_trials`, `rule_correct`, `rule_unreplaced`, `rule_wrong_rule`, `rule_other_word`, `rule_deleted` | the rule words by their outcomes |
| `background_trials`, `background_misfires`, `background_other_change`, `background_deleted` | the background words by their outcomes |
| `pairings_one`, `pairings_glue`, `pairings_split` | the pairings of lines in the gaps |

`scripts/measure_point.py` prints the main counts of the point it measured. The passport uses the
trials of every answered point with their conditions ([passport model](05-passport-model.md)).

## Limitations of the alignment

- The alignment reads the order of the words, not the intention of the model. A line rewritten with
  many wrong words falls under the copy threshold and is counted as a deleted line and an inserted
  line.
- A line with an error in a word that a rule does not cover stands on its text line only while its
  background words keep their places and more than half of its words agree; otherwise it is aligned
  in a gap, and its words are still counted on their places.
- The instrument does not tell a repeat from a return: both are insertions.
