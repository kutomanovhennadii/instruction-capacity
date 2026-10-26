# Answer validation

The answer of the model becomes a sequence of Bernoulli trials and a few facts about its
structure. The reference answer is computed by a program, so no judge, human or model, and no
text-similarity metric enters the measurement.

A bad answer is data, not an error of the instrument: a model that lost half of the replacements
or returned noise gives a valid measurement. Only an inconsistency of the instrument's own inputs
stops validation, and technical failures of the provider never reach it: they are the status
`api_error`.

## The reference answer

Each word of the text that stands on the left side of a rule becomes the right side of that rule;
every other word stays. One pass is enough: the words to replace and the replacements are disjoint,
and the replacements never occur in the text ([task construction](03-task-construction.md)). The
positions that change are checked against the positions marked when the text was built.

## Trials and outcomes

Trials are numbered in reading order. A trial scores 1 if the answer has the expected
replacement at its position and 0 otherwise. Each trial also gets an outcome category:

| Category | What stands at the position | Score |
|---|---|---|
| correct | the expected replacement | 1 |
| not replaced | the original word: the rule was lost | 0 |
| foreign target | the replacement of **another** rule: two rules were mixed up | 0 |
| other word | any other word | 0 |
| unreachable | the position cannot be read: the structure of the answer broke before it | 0 |

The record of a trial keeps its number, the position of its rule in the table, the position of
its word, the expected and the observed word, and the category.

Case matters, and normalization is minimal: only the trailing whitespace of the answer is
removed. `Stone` instead of `stone` is a failed trial.

## The wrapper of an answer

Models often frame the answer, with "Here is the rewritten text:", a code fence or a closing
remark, although the task asks for the rewritten text only. Left as is, an extra line would shift
the line anchors, and a perfect rewrite would score like noise. So the edges of the answer are
stripped, and the stripping is counted as a separate fact. Two independent cues decide what is a
wrapper line:

1. **the alphabet of the protocol**: a useful line holds only lowercase Latin letters and spaces,
   so a line with a capital letter, a digit or punctuation, or an empty line, cannot be useful;
2. **the edge words of the core**: the reference says how the core must begin and end; the
   anchor is the pair "right replacement or original word", so a lost replacement at the edge
   does not disable it.

Both cues are subject to one check: the remaining core must have the reference number of lines.
If neither cue passes it, nothing is stripped. A line of full reference width is never stripped:
words appended by the model are indistinguishable from the text. The middle of the answer is never
touched.

## Lines as synchronization anchors

Compared as one flat stream, a single missing word would shift every later word and zero the rest
of the text. Instead the answer is matched with the reference **line by line**:

1. lines are matched by their number;
2. inside a line, reachability is decided by the **number of words**, not by their content: a line
   with the reference number of words is scored position by position; otherwise the matching
   prefix is kept and the rest of the line is unreachable;
3. the next line is synchronized anew by its own anchor.

A wrong replacement is a measured event, not a loss of synchronization. The damage of a local
break is at most the tail of one line, and the model never scores above what it deserves:
unreachable positions are failures.

## Three kinds of collapse

| Kind | What happened | What is recorded |
|---|---|---|
| first | a wrong replacement with the structure intact | the first failed trial and its rule |
| second | a broken line absorbed by itself: its word count differs, but the next line is recognizable | the first such line |
| third | control over the lines is lost: the damage is not absorbed, or the number of lines differs | the line where the answer loses synchronization with its text |

A broken line is absorbed if the next line has the reference number of words or the broken line
is the last one. A different number of lines is a third-kind collapse on its own, in either
direction. "Kind" here is a scale of damage of the answer; it has nothing to do with type I and
type II errors of hypothesis testing.

The line of a third-kind collapse is found by content, not by number. Answer lines are tied to
reference lines only by exact copies and only forward through the text. While a tied line stands
at its own number, matching by number is right; the first tie with a shifted number means that
lines were dropped or merged before it, and the collapse is placed on the first reference line
after the last tie at its own number. Without such a tie, for a cut answer, a break inside a line
or extra lines at the end, the collapse is placed on the first unabsorbed broken line or on the
first line beyond the end of the shorter side, and the place is never later than these lines.
The scores of the trials do not depend on this place.

Changes of words that no rule covers, with the word count intact, are counted separately as
background corruption; they are neither trials nor collapses.

## Statistics of an answer

| Quantity | Meaning |
|---|---|
| share of right replacements per rule | stability of a rule |
| share of right replacements over all trials | the main quantity |
| first failed trial | the first-kind collapse and the rule that failed |
| line of the first second-kind collapse, line where a third-kind collapse began | where the structure broke first and how badly |
| lines with broken structure | the extent of the damage |
| background corruption | words changed outside the rules |
| wrapper | whether a wrapper was stripped and how many lines at each edge |
| strict match | the answer equals the reference byte for byte |

An absent first error is recorded as absent, not as zero: "no error" and "an error at the very
end" are different observations.

The passport uses the trials of every answered point (rule number, word position, score) and the
line of its first third-kind collapse ([passport model](05-passport-model.md)).

## Limitations of the matching

- A whole inserted or removed line shifts the matching of all later lines; this is recorded as a
  third-kind collapse, honestly but coarsely.
- Two adjacent independent second-kind breaks look like a word moved to the next line and are
  conservatively taken as the third kind.
- A lowercase wrapper together with an error in an edge word of the core is not stripped and
  becomes a third-kind collapse.
